#ifndef HeterogeneousCore_SonicTriton_triton_utils
#define HeterogeneousCore_SonicTriton_triton_utils

#include "FWCore/Utilities/interface/Span.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonException.h"

#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>

#include "grpc_client.h"

namespace triton_utils {
  template <typename C>
  std::string printColl(const C& coll, const std::string& delim = ", ");
  //implemented as a standalone function to avoid repeated specializations for different TritonData types
  template <typename DT>
  bool checkType(inference::DataType dtype) { return false; }
}  // namespace triton_utils

//explicit specializations
//FP16 (half precision) is a special case because no C++ primitive exists, so use uint16_t (e.g. with libminifloat)
template <>
bool triton_utils::checkType<bool>(inference::DataType dtype) { return dtype==inference::DataType::TYPE_BOOL; }
template <>
bool triton_utils::checkType<uint8_t>(inference::DataType dtype) { return dtype==inference::DataType::TYPE_UINT8; }
template <>
bool triton_utils::checkType<uint16_t>(inference::DataType dtype) { return dtype==inference::DataType::TYPE_UINT16 or dtype==inference::DataType::TYPE_FP16; }
template <>
bool triton_utils::checkType<uint32_t>(inference::DataType dtype) { return dtype==inference::DataType::TYPE_UINT32; }
template <>
bool triton_utils::checkType<uint64_t>(inference::DataType dtype) { return dtype==inference::DataType::TYPE_UINT64; }
template <>
bool triton_utils::checkType<int8_t>(inference::DataType dtype) { return dtype==inference::DataType::TYPE_INT8; }
template <>
bool triton_utils::checkType<int16_t>(inference::DataType dtype) { return dtype==inference::DataType::TYPE_INT16; }
template <>
bool triton_utils::checkType<int32_t>(inference::DataType dtype) { return dtype==inference::DataType::TYPE_INT32; }
template <>
bool triton_utils::checkType<int64_t>(inference::DataType dtype) { return dtype==inference::DataType::TYPE_INT64; }
template <>
bool triton_utils::checkType<float>(inference::DataType dtype) { return dtype==inference::DataType::TYPE_FP32; }
template <>
bool triton_utils::checkType<double>(inference::DataType dtype) { return dtype==inference::DataType::TYPE_FP64; }

//helper to turn triton error into exception
//implemented as a macro to avoid constructing the MSG string for successful function calls
#define tritonThrowIfError(X, MSG)                                                                            \
  {                                                                                                           \
    triton::client::Error err = (X);                                                                          \
    if (!err.IsOk())                                                                                          \
      throw TritonException("TritonFailure") << (MSG) << (err.Message().empty() ? "" : ": " + err.Message()); \
  }

extern template std::string triton_utils::printColl(const edm::Span<std::vector<int64_t>::const_iterator>& coll,
                                                    const std::string& delim);
extern template std::string triton_utils::printColl(const std::vector<uint8_t>& coll, const std::string& delim);
extern template std::string triton_utils::printColl(const std::vector<float>& coll, const std::string& delim);
extern template std::string triton_utils::printColl(const std::unordered_set<std::string>& coll,
                                                    const std::string& delim);

#endif
