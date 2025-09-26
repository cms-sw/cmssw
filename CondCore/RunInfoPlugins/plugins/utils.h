#ifndef CondCore_RunInfoPlugins_utils_H
#define CondCore_RunInfoPlugins_utils_H

#include "CondCore/CondDB/interface/Time.h"

namespace lhcInfo {
  inline std::pair<unsigned int, unsigned int> unpack(cond::Time_t since) {
    auto kLowMask = 0XFFFFFFFF;
    auto run = (since >> 32);
    auto lumi = (since & kLowMask);
    return std::make_pair(run, lumi);
  }
}  // namespace lhcInfo

#endif
