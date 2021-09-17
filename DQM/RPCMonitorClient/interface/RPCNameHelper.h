#ifndef RPCNameHelper_H
#define RPCNameHelper_H

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <string>
#include <array>

struct RPCNameHelper {
  static std::string name(const RPCDetId& detId, const bool useRoll);
  static std::string chamberName(const RPCDetId& detId);
  static std::string rollName(const RPCDetId& detId);
  static std::string regionName(const int region);

  static const std::array<std::string, 3> regionNames;
};

#endif
