#ifndef DETECTOR_DESCRIPTION_MUON_NUMBERING_H
#define DETECTOR_DESCRIPTION_MUON_NUMBERING_H

#include "DetectorDescription/DDCMS/interface/ExpandedNodes.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include <string>
#include <unordered_map>

class MuonBaseNumber;

namespace cms {

  using MuonConstants = std::unordered_map<std::string_view, int>;
  
  struct MuonNumbering {
    const MuonBaseNumber geoHistoryToBaseNumber(const cms::ExpandedNodes&) const;
    const int get(const char*) const;
    
    MuonConstants values;
  };
}

#endif
