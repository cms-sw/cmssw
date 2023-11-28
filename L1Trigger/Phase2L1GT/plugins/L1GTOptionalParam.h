#ifndef L1Trigger_Phase2L1GT_L1GTOptionalParam_h
#define L1Trigger_Phase2L1GT_L1GTOptionalParam_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <optional>
#include <functional>

namespace l1t {

  template <typename T, typename K>
  inline std::optional<T> getOptionalParam(const std::string& name,
                                           const edm::ParameterSet& config,
                                           std::function<T(K)> conv) {
    if (config.exists(name)) {
      return std::optional<T>(conv(config.getParameter<K>(name)));
    }
    return std::optional<T>();
  }

  template <typename T>
  inline std::optional<T> getOptionalParam(const std::string& name, const edm::ParameterSet& config) {
    if (config.exists(name)) {
      return std::optional<T>(config.getParameter<T>(name));
    }
    return std::optional<T>();
  }
}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GTOptionalParam_h
