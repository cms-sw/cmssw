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
}  // namespace triton_utils

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
