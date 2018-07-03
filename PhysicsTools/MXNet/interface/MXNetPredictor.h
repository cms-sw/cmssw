#ifndef PHYSICSTOOLS_MXNET_MXNETPREDICTOR_H
#define PHYSICSTOOLS_MXNET_MXNETPREDICTOR_H

#include <string>
#include <vector>
#include <memory>
#include "mxnet/c_predict_api.h"

namespace mxnet{
// Read file to buffer
class BufferFile {
public:

  explicit BufferFile(std::string file_path);
  ~BufferFile() {}

  int GetLength() const {
    return buffer_.length();
  }
  const char* GetBuffer() const {
    return buffer_.c_str();
  }

private:
  std::string file_path_;
  std::string buffer_;
};

class MXNetPredictor {
public:
  MXNetPredictor();
  virtual ~MXNetPredictor();

  void set_input_shapes(const std::vector<std::string>& input_names, const std::vector<std::vector<mx_uint>>& input_shapes);
  void load_model(const BufferFile* model_data, const BufferFile* param_data);
  std::vector<float> predict(const std::vector<std::vector<mx_float>>& input_data, mx_uint output_index = 0);

private:
  mx_uint num_input_nodes_ = 0;
  std::vector<std::string> input_names_;
  std::vector<const char*> input_keys_;

  std::vector<mx_uint> input_shapes_indices_;
  std::vector<mx_uint> input_shapes_data_;

  PredictorHandle pred_hnd_ = nullptr;
};

}

#endif /* PHYSICSTOOLS_MXNET_MXNETPREDICTOR_H */
