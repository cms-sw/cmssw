/*
 * ONNXRuntime.cc
 *
 *  Created on: Jun 28, 2019
 *      Author: hqu
 */

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace cms::Ort {

  using namespace ::Ort;

  namespace {

    inline int64_t numel(const std::vector<int64_t>& dims) {
      return std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
    }

    inline void ensureNoDynamicDimsExceptBatch(const std::vector<int64_t>& dims, const std::string& name) {
      for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == -1 && i != 0) {
          throw cms::Exception("RuntimeError") << "Output " << name << " has a dynamic dimension at index " << i
                                               << ". Please pass output_shapes to runInto().";
        }
      }
    }

  }  // namespace

  const Env ONNXRuntime::env_(ORT_LOGGING_LEVEL_ERROR, "");

  ONNXRuntime::ONNXRuntime(const std::string& model_path, const SessionOptions* session_options) {
    // create session
    if (session_options) {
      session_ = std::make_unique<Session>(env_, model_path.c_str(), *session_options);
    } else {
      session_ = std::make_unique<Session>(env_, model_path.c_str(), defaultSessionOptions());
    }
    AllocatorWithDefaultOptions allocator;

    // get input names and shapes
    size_t num_input_nodes = session_->GetInputCount();
    input_node_strings_.resize(num_input_nodes);
    input_node_names_.resize(num_input_nodes);
    input_node_dims_.clear();

    for (size_t i = 0; i < num_input_nodes; i++) {
      // get input node names
      std::string input_name(session_->GetInputNameAllocated(i, allocator).get());
      input_node_strings_[i] = input_name;
      input_node_names_[i] = input_node_strings_[i].c_str();

      // get input shapes
      auto type_info = session_->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

      input_node_dims_[input_name] = tensor_info.GetShape();
    }

    size_t num_output_nodes = session_->GetOutputCount();
    output_node_strings_.resize(num_output_nodes);
    output_node_names_.resize(num_output_nodes);
    output_node_dims_.clear();

    for (size_t i = 0; i < num_output_nodes; i++) {
      // get output node names
      std::string output_name(session_->GetOutputNameAllocated(i, allocator).get());
      output_node_strings_[i] = output_name;
      output_node_names_[i] = output_node_strings_[i].c_str();

      // get output node types
      auto type_info = session_->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      output_node_dims_[output_name] = tensor_info.GetShape();

      // the 0th dim depends on the batch size
      output_node_dims_[output_name].at(0) = -1;
    }
  }

  ONNXRuntime::~ONNXRuntime() {}

  SessionOptions ONNXRuntime::defaultSessionOptions(Backend backend) {
    SessionOptions sess_opts;
    sess_opts.SetIntraOpNumThreads(1);
    if (backend == Backend::cuda) {
      // https://www.onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html
      OrtCUDAProviderOptions options;
      sess_opts.AppendExecutionProvider_CUDA(options);
    }
    return sess_opts;
  }

  void ONNXRuntime::runInto(const std::vector<std::string>& input_names,
                            FloatArrays& input_values,
                            const std::vector<std::vector<int64_t>>& input_shapes,
                            const std::vector<std::string>& output_names,
                            FloatArrays& output_values,
                            const std::vector<std::vector<int64_t>>& output_shapes,
                            int64_t batch_size) const {
    assert(input_names.size() == input_values.size());
    assert(input_shapes.empty() || input_names.size() == input_shapes.size());
    assert(output_shapes.empty() || (!output_names.empty() && output_shapes.size() == output_names.size()));
    assert(batch_size > 0);

    // Fast lookup input name -> position
    std::unordered_map<std::string, size_t> inputPos;
    inputPos.reserve(input_names.size());
    for (size_t i = 0; i < input_names.size(); ++i) {
      inputPos.emplace(input_names[i], i);
    }

    // create input tensor objects from data values
    std::vector<Value> input_tensors;
    input_tensors.reserve(input_node_strings_.size());

    auto memory_info = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (const auto& name : input_node_strings_) {
      auto it = inputPos.find(name);
      if (it == inputPos.end()) {
        throw cms::Exception("RuntimeError") << "Input " << name << " is not provided!";
      }
      const size_t input_pos = it->second;
      auto& value = input_values[input_pos];

      // Get input dimensions: use provided shapes if available, otherwise fall back to ONNX model defaults
      const auto& onnx_dims = input_node_dims_.at(name);
      std::vector<int64_t> input_dims = input_shapes.empty() ? onnx_dims : input_shapes[input_pos];

      // Check if the model expects a dynamic batch dimension (indicated by -1)
      const bool has_dynamic_batch = !onnx_dims.empty() && (onnx_dims[0] == -1);

      if (has_dynamic_batch) {
        if (input_shapes.empty()) {
          // No shapes provided then enforce the current batch size
          input_dims[0] = batch_size;
        } else if (input_dims[0] != batch_size) {
          // Shapes provided but batch size mismatch then update global batch size
          batch_size = input_dims[0];
        }
      }

      const int64_t expected_len = numel(input_dims);
      if (expected_len != static_cast<int64_t>(value.size())) {
        throw cms::Exception("RuntimeError")
            << "Input array " << name << " has a wrong size of " << value.size() << ", expected " << expected_len;
      }

      auto input_tensor =
          Value::CreateTensor<float>(memory_info, value.data(), value.size(), input_dims.data(), input_dims.size());
      assert(input_tensor.IsTensor());
      input_tensors.emplace_back(std::move(input_tensor));
    }

    // Resolve output node names; will get all outputs if `output_names` is not provided
    std::vector<std::string> resolved_output_names;
    if (output_names.empty()) {
      resolved_output_names = output_node_strings_;
    } else {
      resolved_output_names = output_names;
    }

    std::vector<const char*> run_output_node_names;
    run_output_node_names.reserve(resolved_output_names.size());
    for (const auto& n : resolved_output_names) {
      run_output_node_names.push_back(n.c_str());
    }

    // Prepare output buffers and Ort::Value tensors that write directly into them
    output_values.resize(resolved_output_names.size());

    std::vector<Value> output_tensors;
    output_tensors.reserve(resolved_output_names.size());

    for (size_t i = 0; i < resolved_output_names.size(); ++i) {
      const auto& out_name = resolved_output_names[i];

      std::vector<int64_t> out_dims;
      if (!output_shapes.empty()) {
        out_dims = output_shapes[i];
      } else {
        out_dims = getOutputShape(out_name);
        if (!out_dims.empty() && out_dims[0] == -1) {
          out_dims[0] = batch_size;
        }
        ensureNoDynamicDimsExceptBatch(out_dims, out_name);
      }

      const int64_t out_len = numel(out_dims);
      if (out_len <= 0) {
        throw cms::Exception("RuntimeError") << "Output " << out_name << " has invalid inferred size " << out_len;
      }

      auto& out_buf = output_values[i];
      if (static_cast<int64_t>(out_buf.capacity()) < out_len) {
        out_buf.reserve(static_cast<size_t>(out_len));
      }
      out_buf.resize(static_cast<size_t>(out_len));

      auto out_tensor =
          Value::CreateTensor<float>(memory_info, out_buf.data(), out_buf.size(), out_dims.data(), out_dims.size());
      assert(out_tensor.IsTensor());
      output_tensors.emplace_back(std::move(out_tensor));
    }

    // Run inference writing directly into output_tensors
    session_->Run(RunOptions{nullptr},
                  input_node_names_.data(),
                  input_tensors.data(),
                  input_tensors.size(),
                  run_output_node_names.data(),
                  output_tensors.data(),
                  output_tensors.size());
  }

  FloatArrays ONNXRuntime::run(const std::vector<std::string>& input_names,
                               FloatArrays& input_values,
                               const std::vector<std::vector<int64_t>>& input_shapes,
                               const std::vector<std::string>& output_names,
                               int64_t batch_size) const {
    FloatArrays outputs;
    runInto(input_names, input_values, input_shapes, output_names, outputs, {}, batch_size);
    return outputs;
  }

  const std::vector<std::string>& ONNXRuntime::getOutputNames() const {
    if (session_) {
      return output_node_strings_;
    } else {
      throw cms::Exception("RuntimeError") << "Needs to call createSession() first before getting the output names!";
    }
  }

  const std::vector<int64_t>& ONNXRuntime::getOutputShape(const std::string& output_name) const {
    auto iter = output_node_dims_.find(output_name);
    if (iter == output_node_dims_.end()) {
      throw cms::Exception("RuntimeError") << "Output name " << output_name << " is invalid!";
    } else {
      return iter->second;
    }
  }

} /* namespace cms::Ort */
