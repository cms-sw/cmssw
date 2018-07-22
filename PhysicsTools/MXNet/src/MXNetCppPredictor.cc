/*
 * MXNetCppPredictor.cc
 *
 *  Created on: Jul 19, 2018
 *      Author: hqu
 */

#include <cassert>
#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MXNet/interface/MXNetCppPredictor.h"

namespace mxnet {

namespace cpp {

Block::Block() {
}

Block::Block(const std::string& symbol_file, const std::string& param_file) {
  // load the symbol
  sym_ = Symbol::Load(symbol_file);
  // load the parameters
  load_parameters(param_file);
}

Block::~Block() {
}

void Block::load_parameters(const std::string& param_file) {
  std::map<std::string, NDArray> paramters;
  NDArray::Load(param_file, nullptr, &paramters);
  for (const auto &k : paramters) {
    if (k.first.substr(0, 4) == "aux:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      aux_map_[name] = k.second;
    }
    if (k.first.substr(0, 4) == "arg:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      arg_map_[name] = k.second;
    }
  }
}

const Context MXNetCppPredictor::context_ = Context(DeviceType::kCPU, 0);

MXNetCppPredictor::MXNetCppPredictor() {
}

MXNetCppPredictor::MXNetCppPredictor(const Block& block) : sym_(block.symbol()), arg_map_(block.arg_map()), aux_map_(block.aux_map()) {
}

MXNetCppPredictor::~MXNetCppPredictor() {
}

void MXNetCppPredictor::set_input_shapes(const std::vector<std::string>& input_names, const std::vector<std::vector<mx_uint> >& input_shapes) {
  assert(input_names.size() == input_shapes.size());
  input_names_ = input_names;
  // init the input NDArrays and add them to the arg_map
  for (unsigned i=0; i<input_names_.size(); ++i){
    const auto& name = input_names_.at(i);
    NDArray nd(input_shapes.at(i), context_, false);
    arg_map_[name] = nd;
  }
}

void MXNetCppPredictor::set_output_node_name(const std::string& output_node_name) {
  if (!output_node_name.empty()){
    sym_ = sym_.GetInternals()[output_node_name];
  }
}

const std::vector<float>& MXNetCppPredictor::predict(const std::vector<std::vector<mx_float> >& input_data) {
  assert(input_names_.size() == input_data.size());

  try {
    // create the executor (if not done yet)
    if (!exec_) { bind_executor(); }
    assert(exec_);
    // set the inputs
    for (unsigned i=0; i<input_names_.size(); ++i){
      const auto& name = input_names_.at(i);
      arg_map_[name].SyncCopyFromCPU(input_data.at(i));
    }
    // run forward
    exec_->Forward(false);
    // copy the output to pred_
    exec_->outputs[0].SyncCopyToCPU(&pred_);
    return pred_;
  }catch(const dmlc::Error &e){
    throw cms::Exception("RuntimeError") << e.what() << MXGetLastError();
  }
}

void MXNetCppPredictor::bind_executor() {

  // infer shapes
  const auto arg_name_list = sym_.ListArguments();
  std::vector<std::vector<mx_uint> > in_shapes, aux_shapes, out_shapes;
  std::map<std::string, std::vector<mx_uint> > arg_shapes;

  for (const auto &arg_name : arg_name_list) {
    auto iter = arg_map_.find(arg_name);
    if (iter != arg_map_.end()) {
      arg_shapes[arg_name] = iter->second.GetShape();
    }
  }
  sym_.InferShape(arg_shapes, &in_shapes, &aux_shapes, &out_shapes);

  // init argument arrays
  std::vector<NDArray> arg_arrays;
  for (size_t i = 0; i < in_shapes.size(); ++i) {
    const auto &shape = in_shapes[i];
    const auto &arg_name = arg_name_list[i];
    auto iter_arg = arg_map_.find(arg_name);
    if (iter_arg != arg_map_.end()) {
      arg_arrays.push_back(iter_arg->second);
    } else {
      arg_arrays.push_back(NDArray(shape, context_, false));
    }
  }
  std::vector<NDArray> grad_arrays(arg_arrays.size());
  std::vector<OpReqType> grad_reqs(arg_arrays.size(), kNullOp);

  // init auxiliary array
  std::vector<NDArray> aux_arrays;
  const auto aux_name_list = sym_.ListAuxiliaryStates();
  for (size_t i = 0; i < aux_shapes.size(); ++i) {
    const auto &shape = aux_shapes[i];
    const auto &aux_name = aux_name_list[i];
    auto iter_aux = aux_map_.find(aux_name);
    if (iter_aux != aux_map_.end()) {
      aux_arrays.push_back(iter_aux->second);
    } else {
      aux_arrays.push_back(NDArray(shape, context_, false));
    }
  }

  // bind executor
  exec_.reset(new Executor(sym_, context_, arg_arrays, grad_arrays, grad_reqs, aux_arrays));

}

}

} /* namespace mxnet */

