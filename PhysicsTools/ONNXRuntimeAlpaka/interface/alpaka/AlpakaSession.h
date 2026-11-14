#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_alpaka_AlpakaSession_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_alpaka_AlpakaSession_h

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/onnxruntime_run_options_config_keys.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXInterface.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/Exception.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/GetMemoryInfo.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/SessionOptions.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/alpaka/PackKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ort {

  // ONNX Runtime session running the inference directly on SoA data in alpaka device memory.
  //
  // ONNX Runtime binds a CUDA stream to a session when the session is created, so each AlpakaSession is bound to a
  // single alpaka queue: the one passed to bind(), or, if bind() has not been called, the one passed to the first call
  // to forward(). When forward() is called with a different queue, the inference runs in the bound queue, and is
  // synchronised with the queue passed to forward() using alpaka events, without blocking the host.
  // In a FixedQueueEDProducer, call bind() from beginStream(): the inference then runs directly in the queue used by
  // the module for all events, without any synchronisation.
  // An AlpakaSession is not thread safe, and should be used by a single EDM stream, e.g. as a data member of a
  // stream module.
  class AlpakaSession {
  public:
    explicit AlpakaSession(std::string model_path) : model_path_(std::move(model_path)) {}

    AlpakaSession(AlpakaSession const&) = delete;
    AlpakaSession& operator=(AlpakaSession const&) = delete;

    // Create the ONNX Runtime session, running on the device and in the stream of the given queue.
    void bind(Queue const& queue) {
      using cms::Ort::alpakatools::detail::throwException;
      if (session_) {
        throwException("LogicError", "The ONNX Runtime session for " + model_path_ + " is already bound.");
      }

      try {
        auto options = cms::Ort::alpakatools::sessionOptions(queue);
        session_ = std::make_unique<::Ort::Session>(
            edm::Service<cms::Ort::ONNXInterface>()->environment(), model_path_.c_str(), options);

        ::Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
          input_names_.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
          input_types_.push_back(session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
        }
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
          output_names_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
          output_types_.push_back(session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
        }
      } catch (::Ort::Exception const& e) {
        session_.reset();
        input_names_.clear();
        input_types_.clear();
        output_names_.clear();
        output_types_.clear();
        throwException("ONNXRuntime",
                       "ONNX Runtime error " + std::to_string(e.GetOrtErrorCode()) + ": " + e.what(),
                       "Creating the ONNX Runtime session for the model " + model_path_);
      }
      for (auto const& name : input_names_)
        input_name_ptrs_.push_back(name.c_str());
      for (auto const& name : output_names_)
        output_name_ptrs_.push_back(name.c_str());

      queue_ = queue;
      ready_.emplace(alpaka::getDev(queue));
      done_.emplace(alpaka::getDev(queue));
      memory_info_ = cms::Ort::alpakatools::getMemoryInfo(alpaka::getDev(queue));

      // do not synchronise the device at the end of each inference
      run_options_.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
    }

    bool bound() const { return static_cast<bool>(session_); }

    // Forward pass (inference) of the model with SoA metadata input/output.
    // The tensors are matched to the model inputs and outputs by name if all their names are the names used in the
    // model, otherwise by position (see TensorCollection::match()); all the model outputs must be provided.
    // The outputs are written directly into the SoA memory.
    // If any input tensor is empty, the inference is skipped, and the outputs are filled with zeros.
    void forward(Queue& queue,
                 cms::Ort::alpakatools::TensorCollection<Queue> const& inputs,
                 cms::Ort::alpakatools::TensorCollection<Queue> const& outputs) {
      if (not session_) {
        bind(queue);
      }
      const auto input_index = inputs.match(input_names_, "input", model_path_);
      const auto output_index = outputs.match(output_names_, "output", model_path_);
      check(queue, inputs, input_index, outputs, output_index);

      // ONNX Runtime does not support empty tensors reliably: if any input is empty, skip the inference and fill
      // the outputs with zeros, in the event queue
      for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].batch_size() == 0) {
          for (size_t j = 0; j < outputs.size(); ++j)
            zero(queue, outputs[j]);
          return;
        }
      }

      // synchronise the queue used by the session with the event queue
      Queue& run_queue = *queue_;
      const bool bridge = not(queue == run_queue);
      if (bridge) {
        alpaka::enqueue(queue, *ready_);
        alpaka::wait(run_queue, *ready_);
      }

      // wrap the SoA memory in ONNX Runtime tensors, in the order of the model inputs and outputs, packing it into
      // temporary buffers if needed
      std::vector<Staging> staging(inputs.size() + outputs.size());
      std::vector<::Ort::Value> input_values;
      input_values.reserve(inputs.size());
      for (size_t i = 0; i < inputs.size(); ++i) {
        input_values.push_back(stage(run_queue, inputs[input_index[i]], true, staging[i]));
      }
      std::vector<::Ort::Value> output_values;
      output_values.reserve(outputs.size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        output_values.push_back(stage(run_queue, outputs[output_index[i]], false, staging[inputs.size() + i]));
      }

      try {
        session_->Run(run_options_,
                      input_name_ptrs_.data(),
                      input_values.data(),
                      input_values.size(),
                      output_name_ptrs_.data(),
                      output_values.data(),
                      output_values.size());
      } catch (::Ort::Exception const& e) {
        cms::Ort::alpakatools::detail::throwException("ONNXRuntime",
                                                      "ONNX Runtime error " + std::to_string(e.GetOrtErrorCode()) +
                                                          ": " + e.what() +
                                                          "\nInputs:" + describe(inputs, input_index, input_names_) +
                                                          "\nOutputs:" + describe(outputs, output_index, output_names_),
                                                      "Running the inference of the model " + model_path_);
      }

      // unpack the outputs into the SoA memory, if needed
      for (size_t i = 0; i < outputs.size(); ++i) {
        unstage(run_queue, outputs[output_index[i]], staging[inputs.size() + i]);
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
    // temporary buffer used to present a tensor to ONNX Runtime
    struct Staging {
      std::optional<cms::alpakatools::device_buffer<Device, std::byte[]>> device;  // packed tensor
    };

    void check(Queue const& queue,
               cms::Ort::alpakatools::TensorCollection<Queue> const& inputs,
               std::vector<size_t> const& input_index,
               cms::Ort::alpakatools::TensorCollection<Queue> const& outputs,
               std::vector<size_t> const& output_index) const {
      using cms::Ort::alpakatools::detail::throwException;
      if (alpaka::getDev(queue) != alpaka::getDev(*queue_)) {
        throwException("LogicError",
                       "The ONNX Runtime session for " + model_path_ +
                           " cannot be used on a different device than the one it is bound to.");
      }
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto const& tensor = inputs[input_index[i]];
        if (tensor.type() != input_types_[i]) {
          throwException("InvalidArgument",
                         "The type of the input tensor '" + inputs.name(input_index[i]) + "' (" +
                             typeName(tensor.type()) + ") does not match the type of the input '" + input_names_[i] +
                             "' of the model " + model_path_ + " (" + typeName(input_types_[i]) + ").");
        }
      }
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto const& tensor = outputs[output_index[i]];
        if (tensor.type() != output_types_[i]) {
          throwException("InvalidArgument",
                         "The type of the output tensor '" + outputs.name(output_index[i]) + "' (" +
                             typeName(tensor.type()) + ") does not match the type of the output '" + output_names_[i] +
                             "' of the model " + model_path_ + " (" + typeName(output_types_[i]) + ").");
        }
        if (tensor.is_scalar()) {
          throwException(
              "InvalidArgument",
              "The output tensor '" + outputs.name(output_index[i]) + "' is a scalar, which is not supported.");
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
      return ::Ort::Value::CreateTensor(memory_info_, data, size, shape.data(), shape.size(), tensor.type());
    }

    void unstage(Queue& queue, cms::Ort::alpakatools::detail::ITensorHandle const& tensor, Staging& staging) const {
      if (not tensor.is_contiguous()) {
        detail::unpack(queue, tensor.pack_descriptor(), staging.device->data(), static_cast<std::byte*>(tensor.data()));
      }
    }

    // fill the memory of an output tensor with zeros
    static void zero(Queue& queue, cms::Ort::alpakatools::detail::ITensorHandle const& tensor) {
      if (tensor.batch_size() == 0)
        return;
      if (tensor.is_contiguous()) {
        auto view = cms::alpakatools::make_device_view(
            queue, static_cast<std::byte*>(tensor.data()), static_cast<cms::alpakatools::Extent>(tensor.shape_bytes()));
        alpaka::memset(queue, view, 0);
      } else {
        detail::zero(queue, tensor.pack_descriptor(), static_cast<std::byte*>(tensor.data()));
      }
    }

    // describe the tensors passed to the model, for the error messages
    static std::string describe(cms::Ort::alpakatools::TensorCollection<Queue> const& tensors,
                                std::vector<size_t> const& index,
                                std::vector<std::string> const& names) {
      std::string result;
      for (size_t i = 0; i < index.size(); ++i) {
        auto const& tensor = tensors[index[i]];
        result += "\n  '" + names[i] + "': tensor '" + tensors.name(index[i]) + "', " + typeName(tensor.type()) + " [";
        auto shape = tensor.shape();
        for (size_t d = 0; d < shape.size(); ++d)
          result += (d == 0 ? "" : ", ") + std::to_string(shape[d]);
        result += ']';
        if (tensor.is_scalar())
          result += ", scalar";
        else if (tensor.layout() == cms::Ort::alpakatools::Layout::FeatureMajor)
          result += ", feature-major";
        else if (not tensor.is_contiguous())
          result += ", packed";
      }
      return result;
    }

    static std::string typeName(ONNXTensorElementDataType type) {
      switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
          return "float";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
          return "double";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
          return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
          return "bfloat16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
          return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
          return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
          return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
          return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
          return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
          return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
          return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
          return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
          return "bool";
        default:
          return "type " + std::to_string(type);
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
