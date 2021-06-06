#ifndef HeterogeneousCore_SonicTriton_triton_utils
#define HeterogeneousCore_SonicTriton_triton_utils

#include "FWCore/Utilities/interface/Span.h"

#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>

#include "grpc_client.h"

namespace triton_utils {

  using Error = nvidia::inferenceserver::client::Error;

  template <typename C>
  std::string printColl(const C& coll, const std::string& delim = ", ");

  //helper to turn triton error into exception
  void throwIfError(const Error& err, std::string_view msg);

  //helper to turn triton error into warning
  bool warnIfError(const Error& err, std::string_view msg);

}  // namespace triton_utils

extern template std::string triton_utils::printColl(const edm::Span<std::vector<int64_t>::const_iterator>& coll,
                                                    const std::string& delim);
extern template std::string triton_utils::printColl(const std::vector<uint8_t>& coll, const std::string& delim);
extern template std::string triton_utils::printColl(const std::vector<float>& coll, const std::string& delim);
extern template std::string triton_utils::printColl(const std::unordered_set<std::string>& coll,
                                                    const std::string& delim);

#endif
