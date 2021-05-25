#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Likely.h"

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
      throw cms::Exception("TritonFailure") << msg << (err.Message().empty() ? "" : ": " + err.Message());
  }

  bool warnIfError(const Error& err, std::string_view msg) {
    if (!err.IsOk())
      edm::LogWarning("TritonWarning") << msg << (err.Message().empty() ? "" : ": " + err.Message());
    return err.IsOk();
  }

  bool warnOrThrowIfError(const Error& err, std::string_view msg, bool canThrow) {
    if (canThrow)
      throwIfError(err, msg);
    return !canThrow ? warnIfError(err, msg) : err.IsOk();
  }

  void warnOrThrow(std::string_view msg, bool canThrow) {
    warnOrThrowIfError(Error("client-side problem"), msg, canThrow);
  }

#ifdef TRITON_ENABLE_GPU
  bool cudaCheck(cudaError_t result, std::string_view msg, bool canThrow) {
    if (LIKELY(result == cudaSuccess))
      return true;

    std::string cudaMsg(std::string(cudaGetErrorName(result)) + ": " + cudaGetErrorString(result));
    warnOrThrowIfError(Error(cudaMsg), msg, canThrow);
    return false;
  }
#endif
}  // namespace triton_utils

template std::string triton_utils::printColl(const edm::Span<std::vector<int64_t>::const_iterator>& coll,
                                             const std::string& delim);
template std::string triton_utils::printColl(const std::vector<uint8_t>& coll, const std::string& delim);
template std::string triton_utils::printColl(const std::vector<float>& coll, const std::string& delim);
template std::string triton_utils::printColl(const std::unordered_set<std::string>& coll, const std::string& delim);
