#include "PhysicsTools/MXNet/interface/MXNetPredictor.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace mxnet;

BufferFile::BufferFile(std::string file_path) : file_path_(file_path) {
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    throw cms::Exception("InvalidArgument") << "Can't open the file. Please check " << file_path << ". \n";
    return;
  }

  buffer_.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
  edm::LogInfo("NNKit") << file_path.c_str() << " ... "<< buffer_.length() << " bytes";
  ifs.close();
}

std::mutex MXNetPredictor::mutex_;

MXNetPredictor::MXNetPredictor() {
}

MXNetPredictor::~MXNetPredictor() {
  if (pred_hnd_){
    // Release Predictor
    MXPredFree(pred_hnd_);
  }
}

void MXNetPredictor::set_input_shapes(const std::vector<std::string>& input_names, const std::vector<std::vector<mx_uint> >& input_shapes) {
  assert(input_names.size() == input_shapes.size());

  input_names_ = input_names;
  num_input_nodes_ = input_names_.size();
  for (const auto &name : input_names_){
    input_keys_.push_back(name.c_str());
  }

  input_shapes_data_.clear();
  input_shapes_indices_ = {0};
  unsigned pos = 0;
  for (const auto &shape : input_shapes){
    pos += shape.size();
    input_shapes_indices_.push_back(pos);
    input_shapes_data_.insert(input_shapes_data_.end(), shape.begin(), shape.end());
  }
}

void MXNetPredictor::load_model(const BufferFile* model_data, const BufferFile* param_data) {
  if (model_data->GetLength() == 0 ||
      param_data->GetLength() == 0) {
    throw cms::Exception("InvalidArgument") << "Invalid input";
  }

  std::stringstream msg;
  msg << "input_shapes_indices_:\n";
  for (const auto &d : input_shapes_indices_) {
    msg << d << ",";
  }
  msg << "\ninput_shapes_data_:\n";
  for (const auto &d : input_shapes_data_) {
    msg << d << ",";
  }
  edm::LogInfo("NNKit") << msg.str();

  // Create Predictor
  int dev_type = 1;  // 1: cpu, 2: gpu
  int dev_id = 0;  // arbitrary.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    int status = MXPredCreate(model_data->GetBuffer(),
        param_data->GetBuffer(),
        param_data->GetLength(),
        dev_type,
        dev_id,
        num_input_nodes_,
        (const char**) input_keys_.data(),
        input_shapes_indices_.data(),
        input_shapes_data_.data(),
        &pred_hnd_);
    if (status != 0){
      throw cms::Exception("RuntimeError") << "Cannot create predictor! " << MXGetLastError();
    }

  }
}

std::vector<float> MXNetPredictor::predict(const std::vector<std::vector<mx_float> >& input_data, mx_uint output_index) {
  assert(input_data.size() == input_names_.size());
  assert(pred_hnd_);

  // set input data
  for (unsigned i=0; i<input_names_.size(); ++i){
    if (MXPredSetInput(pred_hnd_, input_names_.at(i).data(), input_data.at(i).data(), input_data.at(i).size()) != 0){
      throw cms::Exception("RuntimeError") << "Cannot set input " << input_names_.at(i) << "! " << MXGetLastError();
    }
  }

  // Do Predict Forward
  if (MXPredForward(pred_hnd_) != 0){
    throw cms::Exception("RuntimeError") << "Error running forward! " << MXGetLastError();
  }

  mx_uint *shape = nullptr;
  mx_uint shape_len = 0;

  // Get Output Result
  if (MXPredGetOutputShape(pred_hnd_, output_index, &shape, &shape_len) != 0){
    throw cms::Exception("RuntimeError") << "Error getting output shape! " << MXGetLastError();
  }

  size_t size = 1;
  for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

  std::vector<float> prediction(size);

  if (MXPredGetOutput(pred_hnd_, output_index, &(prediction[0]), size) != 0){
    throw cms::Exception("RuntimeError") << "Error getting output values! " << MXGetLastError();
  }

  return prediction;
}
