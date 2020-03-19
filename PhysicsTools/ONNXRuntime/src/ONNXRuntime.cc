/*
 * ONNXRuntime.cc
 *
 *  Created on: Jun 28, 2019
 *      Author: hqu
 */

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace cms::Ort {

  using namespace ::Ort;

  const Env ONNXRuntime::env_(ORT_LOGGING_LEVEL_WARNING, "");

  ONNXRuntime::ONNXRuntime(const std::string& model_path, const SessionOptions* session_options) {
    // create session
    if (session_options) {
      session_.reset(new Session(env_, model_path.c_str(), *session_options));
    } else {
      SessionOptions sess_opts;
      sess_opts.SetIntraOpNumThreads(1);
      session_.reset(new Session(env_, model_path.c_str(), sess_opts));
    }
    AllocatorWithDefaultOptions allocator;

    // get input names and shapes
    size_t num_input_nodes = session_->GetInputCount();
    input_node_strings_.resize(num_input_nodes);
    input_node_names_.resize(num_input_nodes);
    input_node_dims_.clear();

    for (size_t i = 0; i < num_input_nodes; i++) {
      // get input node names
      std::string input_name(session_->GetInputName(i, allocator));
      input_node_strings_[i] = input_name;
      input_node_names_[i] = input_node_strings_[i].c_str();

      // get input shapes
      auto type_info = session_->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      size_t num_dims = tensor_info.GetDimensionsCount();
      input_node_dims_[input_name].resize(num_dims);
      tensor_info.GetDimensions(input_node_dims_[input_name].data(), num_dims);

      // set the batch size to 1 by default
      input_node_dims_[input_name].at(0) = 1;
    }

    size_t num_output_nodes = session_->GetOutputCount();
    output_node_strings_.resize(num_output_nodes);
    output_node_names_.resize(num_output_nodes);
    output_node_dims_.clear();

    for (size_t i = 0; i < num_output_nodes; i++) {
      // get output node names
      std::string output_name(session_->GetOutputName(i, allocator));
      output_node_strings_[i] = output_name;
      output_node_names_[i] = output_node_strings_[i].c_str();

      // get output node types
      auto type_info = session_->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      size_t num_dims = tensor_info.GetDimensionsCount();
      output_node_dims_[output_name].resize(num_dims);
      tensor_info.GetDimensions(output_node_dims_[output_name].data(), num_dims);

      // the 0th dim depends on the batch size
      output_node_dims_[output_name].at(0) = -1;
    }
  }

  ONNXRuntime::~ONNXRuntime() {}

  FloatArrays ONNXRuntime::run(const std::vector<std::string>& input_names,
                               FloatArrays& input_values,
                               const std::vector<std::string>& output_names,
                               int64_t batch_size) const {
    assert(input_names.size() == input_values.size());
    assert(batch_size > 0);

    // create input tensor objects from data values
    std::vector<Value> input_tensors;
    auto memory_info = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (const auto& name : input_node_strings_) {
      auto iter = std::find(input_names.begin(), input_names.end(), name);
      if (iter == input_names.end()) {
        throw cms::Exception("RuntimeError") << "Input " << name << " is not provided!";
      }
      auto value = input_values.begin() + (iter - input_names.begin());
      auto input_dims = input_node_dims_.at(name);
      input_dims[0] = batch_size;
      auto expected_len = std::accumulate(input_dims.begin(), input_dims.end(), 1, std::multiplies<int64_t>());
      if (expected_len != (int64_t)value->size()) {
        throw cms::Exception("RuntimeError")
            << "Input array " << name << " has a wrong size of " << value->size() << ", expected " << expected_len;
      }
      auto input_tensor =
          Value::CreateTensor<float>(memory_info, value->data(), value->size(), input_dims.data(), input_dims.size());
      assert(input_tensor.IsTensor());
      input_tensors.emplace_back(std::move(input_tensor));
    }

    // set output node names; will get all outputs if `output_names` is not provided
    std::vector<const char*> run_output_node_names;
    if (output_names.empty()) {
      run_output_node_names = output_node_names_;
    } else {
      for (const auto& name : output_names) {
        run_output_node_names.push_back(name.c_str());
      }
    }

    // run
    auto output_tensors = session_->Run(RunOptions{nullptr},
                                        input_node_names_.data(),
                                        input_tensors.data(),
                                        input_tensors.size(),
                                        run_output_node_names.data(),
                                        run_output_node_names.size());

    // convert output to floats
    FloatArrays outputs;
    for (auto& output_tensor : output_tensors) {
      assert(output_tensor.IsTensor());

      // get output shape
      auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
      auto length = tensor_info.GetElementCount();

      auto floatarr = output_tensor.GetTensorMutableData<float>();
      outputs.emplace_back(floatarr, floatarr + length);
    }
    assert(outputs.size() == run_output_node_names.size());

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
