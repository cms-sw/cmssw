#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <sstream>
#include <experimental/iterator>

namespace triton_utils {

  template <typename C>
  std::string printColl(const C& coll, const std::string& delim) {
    if (coll.empty())
      return "";
    std::stringstream msg;
    //avoid trailing delim
    std::copy(std::begin(coll), std::end(coll), std::experimental::make_ostream_joiner(msg, delim));
    return msg.str();
  }

  void throwIfError(const Error& err, std::string_view msg) {
    if (!err.IsOk())
      throw cms::Exception("TritonServerFailure") << msg << ": " << err;
  }

  bool warnIfError(const Error& err, std::string_view msg) {
    if (!err.IsOk())
      edm::LogWarning("TritonServerWarning") << msg << ": " << err;
    return err.IsOk();
  }

}  // namespace triton_utils

template std::string triton_utils::printColl(const edm::Span<std::vector<int64_t>::const_iterator>& coll,
                                             const std::string& delim);
template std::string triton_utils::printColl(const std::vector<uint8_t>& coll, const std::string& delim);
template std::string triton_utils::printColl(const std::vector<float>& coll, const std::string& delim);
template std::string triton_utils::printColl(const std::unordered_set<std::string>& coll, const std::string& delim);
