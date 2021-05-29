#include "DQM/RPCMonitorClient/interface/RPCNameHelper.h"
#include <fmt/format.h>

const std::array<std::string, 3> RPCNameHelper::regionNames = {{"Endcap-", "Barrel", "Endcap+"}};

std::string RPCNameHelper::name(const RPCDetId& detId, const bool useRoll) {
  return useRoll ? rollName(detId) : chamberName(detId);
}

std::string RPCNameHelper::rollName(const RPCDetId& detId) {
  std::string chName = chamberName(detId);
  const int region = detId.region();
  const int roll = detId.roll();

  if (region == 0) {
    if (roll == 1)
      chName += "_Backward";
    else if (roll == 3)
      chName += "_Forward";
    else
      chName += "_Middle";
  } else {
    if (roll == 1)
      chName += "_A";
    else if (roll == 2)
      chName += "_B";
    else if (roll == 3)
      chName += "_C";
    else if (roll == 4)
      chName += "_D";
    else if (roll == 5)
      chName += "_E";
  }

  return chName;
}

std::string RPCNameHelper::chamberName(const RPCDetId& detId) {
  const int region = detId.region();
  const int sector = detId.sector();
  if (region != 0) {
    // Endcap
    const int disk = detId.region() * detId.station();
    const int ring = detId.ring();
    const int nsub = (ring == 1 and detId.station() > 1) ? 3 : 6;
    const int segment = detId.subsector() + (detId.sector() - 1) * nsub;

    return fmt::format("RE{:+2d}_R{}_CH{:02d}", disk, ring, segment);
  } else {
    // Barrel
    const int wheel = detId.ring();
    const int station = detId.station();
    const int layer = detId.layer();
    const int subsector = detId.subsector();

    std::string roll;
    if (station <= 2) {
      roll = (layer == 1) ? "in" : "out";
    } else if (station == 3) {
      roll = (subsector == 1) ? "-" : "+";
    } else {  // station == 4
      if (sector == 4) {
        const static std::array<std::string, 4> ssarr = {{"--", "-", "+", "++"}};
        roll = ssarr[subsector - 1];
      } else if (sector != 9 && sector != 11) {
        roll = (subsector == 1) ? "-" : "+";
      }
    }

    return fmt::format("W{:+2d}_RB{:d}{}_S{:02d}", wheel, station, roll, sector);
  }

  return "";
}

std::string RPCNameHelper::regionName(const int region) {
  if (region < -1 or region > 1)
    return "";
  return regionNames[region + 1];
}
