/*
 * MXNetCppPredictor.h
 *
 *  Created on: Jul 19, 2018
 *      Author: hqu
 */

#ifndef PHYSICSTOOLS_MXNET_MXNETCPPPREDICTOR_H_
#define PHYSICSTOOLS_MXNET_MXNETCPPPREDICTOR_H_

#include <map>
#include <vector>
#include <memory>
#include <mutex>

#include "mxnet-cpp/MxNetCpp.h"

namespace mxnet {

namespace cpp {

// note: Most of the objects in mxnet::cpp are effective just shared_ptr's

// Simple class to hold MXNet model (symbol + params)
// designed to be sharable by multiple threads
class Block {
public:
  Block();
  Block(const std::string &symbol_file, const std::string &param_file);
  virtual ~Block();

  const Symbol& symbol() const { return sym_; }
  Symbol symbol(const std::string &output_node) const { return sym_.GetInternals()[output_node]; }
  const std::map<std::string, NDArray>& arg_map() const { return arg_map_; }
  const std::map<std::string, NDArray>& aux_map() const { return aux_map_; }

private:
  void load_parameters(const std::string& param_file);

  // symbol
  Symbol sym_;
  // argument arrays
  std::map<std::string, NDArray> arg_map_;
  // auxiliary arrays
  std::map<std::string, NDArray> aux_map_;
};

// Simple helper class to run prediction
// this cannot be shared between threads
class Predictor {
public:
  Predictor();
  Predictor(const Block &block);
  Predictor(const Block &block, const std::string &output_node);
  virtual ~Predictor();

  // set input array shapes
  void set_input_shapes(const std::vector<std::string>& input_names, const std::vector<std::vector<mx_uint>>& input_shapes);

  // run prediction
  const std::vector<float>& predict(const std::vector<std::vector<mx_float>>& input_data);

private:
  static std::mutex mutex_;

  void bind_executor();

  // context
  static const Context context_;
  // executor
  std::unique_ptr<Executor> exec_;
  // symbol
  Symbol sym_;
  // argument arrays
  std::map<std::string, NDArray> arg_map_;
  // auxiliary arrays
  std::map<std::string, NDArray> aux_map_;
  // output of the prediction
  std::vector<float> pred_;
  // names of the input nodes
  std::vector<std::string> input_names_;

};

} /* namespace cpp */
} /* namespace mxnet */

#endif /* PHYSICSTOOLS_MXNET_MXNETCPPPREDICTOR_H_ */
