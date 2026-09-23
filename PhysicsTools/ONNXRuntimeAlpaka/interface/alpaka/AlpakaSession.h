#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_alpaka_AlpakaSession_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_alpaka_AlpakaSession_h

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "onnxruntime/onnxruntime_cxx_api.h"
#include "onnxruntime/onnxruntime_run_options_config_keys.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/GetMemoryInfo.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/OrtEnvironment.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/SessionOptions.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/alpaka/PackKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ort {

  // ONNX Runtime session running the inference directly on SoA data in alpaka device memory.
  //
  // ONNX Runtime binds a CUDA stream to a session when the session is created, so each AlpakaSession is bound to a
  // single alpaka queue: the one passed to bind(), or to the first call to forward().
  //   - In a FixedQueueEDProducer, call bind() from beginStream(): the inference then runs directly in the queue used
  //     by the module for all events.
  //   - In other modules, the queue changes from event to event: the inference runs in the bound queue, and is
  //     synchronised with the event queue using alpaka events, without blocking the host.
  // An AlpakaSession is not thread safe, and should be used by a single EDM stream, e.g. as a data member of a
  // stream module.
  class AlpakaSession {
  public:
    explicit AlpakaSession(std::string model_path) : model_path_(std::move(model_path)) {}

    AlpakaSession(AlpakaSession const&) = delete;
    AlpakaSession& operator=(AlpakaSession const&) = delete;

    // Create the ONNX Runtime session, running on the device and in the stream of the given queue.
    void bind(Queue const& queue) {
      if (session_) {
        throw cms::Exception("LogicError") << "The ONNX Runtime session for " << model_path_ << " is already bound.";
      }
      queue_ = queue;
      ready_.emplace(alpaka::getDev(queue));
      done_.emplace(alpaka::getDev(queue));
      memory_info_ = cms::Ort::alpakatools::getMemoryInfo(alpaka::getDev(queue));

      ::Ort::Env& env = cms::Ort::alpakatools::environment();
      auto options = cms::Ort::alpakatools::sessionOptions(queue);
      session_ = std::make_unique<::Ort::Session>(env, model_path_.c_str(), options);

      // do not synchronise the device at the end of each inference
      run_options_.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");

      ::Ort::AllocatorWithDefaultOptions allocator;
      for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        input_names_.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
        input_types_.push_back(session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
      }
      for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        output_names_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
        output_types_.push_back(session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
      }
      for (auto const& name : input_names_)
        input_name_ptrs_.push_back(name.c_str());
      for (auto const& name : output_names_)
        output_name_ptrs_.push_back(name.c_str());
    }

    bool bound() const { return static_cast<bool>(session_); }

    // Forward pass (inference) of the model with SoA metadata input/output.
    // The inputs and outputs are matched to the model inputs and outputs by position; all the model outputs must be
    // provided. The outputs are written directly into the SoA memory.
    void forward(Queue& queue,
                 cms::Ort::alpakatools::TensorCollection<Queue>& inputs,
                 cms::Ort::alpakatools::TensorCollection<Queue>& outputs) {
      if (not session_) {
        bind(queue);
      }
      check(queue, inputs, outputs);

      // ONNX Runtime does not support empty tensors reliably, and there is nothing to compute anyway
      for (size_t i = 0; i < inputs.size(); ++i)
        if (inputs[i].batch_size() == 0)
          return;

      // synchronise the queue used by the session with the event queue
      Queue& run_queue = *queue_;
      const bool bridge = not(queue == run_queue);
      if (bridge) {
        alpaka::enqueue(queue, *ready_);
        alpaka::wait(run_queue, *ready_);
      }

      // wrap the SoA memory in ONNX Runtime tensors, packing it into temporary buffers if needed
      std::vector<Staging> staging(inputs.size() + outputs.size());
      std::vector<::Ort::Value> input_values;
      input_values.reserve(inputs.size());
      for (size_t i = 0; i < inputs.size(); ++i) {
        input_values.push_back(stage(run_queue, inputs[i], true, staging[i]));
      }
      std::vector<::Ort::Value> output_values;
      output_values.reserve(outputs.size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        output_values.push_back(stage(run_queue, outputs[i], false, staging[inputs.size() + i]));
      }
      if constexpr (not kNative) {
        // the inference runs on the CPU: wait for the inputs to be copied to the host
        alpaka::wait(run_queue);
      }

      session_->Run(run_options_,
                    input_name_ptrs_.data(),
                    input_values.data(),
                    input_values.size(),
                    output_name_ptrs_.data(),
                    output_values.data(),
                    output_values.size());

      // copy the outputs back to the device and unpack them into the SoA memory, if needed
      for (size_t i = 0; i < outputs.size(); ++i) {
        unstage(run_queue, outputs[i], staging[inputs.size() + i]);
      }

      // let the event queue wait for the inference to complete
      if (bridge) {
        alpaka::enqueue(run_queue, *done_);
        alpaka::wait(queue, *done_);
      }
    }

    std::vector<std::string> const& inputNames() const { return input_names_; }
    std::vector<std::string> const& outputNames() const { return output_names_; }

  private:
    // true if ONNX Runtime can access the device memory directly, false if the inference falls back to the CPU
    static constexpr bool kNative = cms::Ort::alpakatools::isNativeDevice<Device>();

    // temporary buffers used to present a tensor to ONNX Runtime
    struct Staging {
      std::optional<cms::alpakatools::device_buffer<Device, std::byte[]>> device;  // packed tensor
      std::optional<cms::alpakatools::host_buffer<std::byte[]>> host;              // CPU fallback
    };

    void check(Queue const& queue,
               cms::Ort::alpakatools::TensorCollection<Queue> const& inputs,
               cms::Ort::alpakatools::TensorCollection<Queue> const& outputs) const {
      if (alpaka::getDev(queue) != alpaka::getDev(*queue_)) {
        throw cms::Exception("LogicError") << "The ONNX Runtime session for " << model_path_
                                           << " cannot be used on a different device than the one it is bound to.";
      }
      if (inputs.size() != input_names_.size() or outputs.size() != output_names_.size()) {
        throw cms::Exception("InvalidArgument")
            << "The model " << model_path_ << " has " << input_names_.size() << " inputs and " << output_names_.size()
            << " outputs, but " << inputs.size() << " inputs and " << outputs.size() << " outputs were provided.";
      }
      for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].type() != input_types_[i]) {
          throw cms::Exception("InvalidArgument")
              << "The type of the input tensor \"" << inputs.name(i) << "\" (" << inputs[i].type()
              << ") does not match the type of the model input \"" << input_names_[i] << "\" (" << input_types_[i]
              << ").";
        }
      }
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].type() != output_types_[i]) {
          throw cms::Exception("InvalidArgument")
              << "The type of the output tensor \"" << outputs.name(i) << "\" (" << outputs[i].type()
              << ") does not match the type of the model output \"" << output_names_[i] << "\" (" << output_types_[i]
              << ").";
        }
        if (outputs[i].is_scalar()) {
          throw cms::Exception("InvalidArgument")
              << "The output tensor \"" << outputs.name(i) << "\" is a scalar, which is not supported.";
        }
      }
    }

    ::Ort::Value stage(Queue& queue,
                       cms::Ort::alpakatools::detail::ITensorHandle const& tensor,
                       bool is_input,
                       Staging& staging) const {
      std::byte* data = static_cast<std::byte*>(tensor.data());
      const auto shape = tensor.shape();
      const size_t size = tensor.shape_bytes();
      if (not tensor.is_contiguous()) {
        staging.device = cms::alpakatools::make_device_buffer<std::byte[]>(queue, size);
        if (is_input)
          detail::pack(queue, tensor.pack_descriptor(), data, staging.device->data());
        data = staging.device->data();
      }
      if constexpr (not kNative) {
        staging.host = cms::alpakatools::make_host_buffer<std::byte[]>(queue, size);
        if (is_input)
          alpaka::memcpy(queue,
                         *staging.host,
                         alpaka::createView(alpaka::getDev(queue), data, alpaka::getExtents(*staging.host)[0]));
        data = staging.host->data();
      }
      return ::Ort::Value::CreateTensor(memory_info_, data, size, shape.data(), shape.size(), tensor.type());
    }

    void unstage(Queue& queue, cms::Ort::alpakatools::detail::ITensorHandle const& tensor, Staging& staging) const {
      std::byte* data = static_cast<std::byte*>(tensor.data());
      std::byte* staged = tensor.is_contiguous() ? data : staging.device->data();
      if constexpr (not kNative) {
        alpaka::memcpy(queue,
                       alpaka::createView(alpaka::getDev(queue), staged, alpaka::getExtents(*staging.host)[0]),
                       *staging.host);
      }
      if (not tensor.is_contiguous()) {
        detail::unpack(queue, tensor.pack_descriptor(), staged, data);
      }
    }

    std::string model_path_;
    std::optional<Queue> queue_;
    std::optional<alpaka::Event<Queue>> ready_;
    std::optional<alpaka::Event<Queue>> done_;
    ::Ort::MemoryInfo memory_info_{nullptr};
    std::unique_ptr<::Ort::Session> session_;
    ::Ort::RunOptions run_options_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
    std::vector<ONNXTensorElementDataType> input_types_;
    std::vector<ONNXTensorElementDataType> output_types_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ort

#endif  // PhysicsTools_ONNXRuntimeAlpaka_interface_alpaka_AlpakaSession_h
