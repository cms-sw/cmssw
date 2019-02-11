#ifndef DETECTOR_DESCRIPTION_MUON_NUMBERING_H
#define DETECTOR_DESCRIPTION_MUON_NUMBERING_H

#include "DetectorDescription/DDCMS/interface/DDExpandedNode.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include <string>
#include <unordered_map>

class MuonBaseNumber;

namespace cms {

  using DDGeoHistory = std::vector<DDExpandedNode>;
  using MuonConstants = std::unordered_map<std::string, int>;
  
  struct MuonNumbering {
    const MuonBaseNumber geoHistoryToBaseNumber(const DDGeoHistory &, MuonConstants&) const;

    MuonConstants values;
  };
}

#endif
