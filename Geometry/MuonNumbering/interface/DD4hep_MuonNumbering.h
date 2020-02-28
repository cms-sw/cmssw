#ifndef GEOMETRY_MUON_NUMBERING_MUON_NUMBERING_H
#define GEOMETRY_MUON_NUMBERING_MUON_NUMBERING_H

// -*- C++ -*-
//
// Package:    Geometry/MuonNumbering
// Class:      MuonNumbering
//
/**\class MuonNumbering

 Description: MuonNumbering class to handle the conversion
 to MuonBaseNumber from the ExpandedNodes history

 Implementation:
 in the xml muon constant section one has to define
 level, super and base constants (eg. 1000,100,1) and
 the start value of the copy numbers (0 or 1)

*/
//
// Original Author:  Ianna Osborne
//         Created:  Thu, 21 Mar 2019 15:32:36 CET
//

#include "DetectorDescription/DDCMS/interface/ExpandedNodes.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"

#include <string>
#include <unordered_map>

class MuonBaseNumber;

namespace cms {

  using MuonConstants = std::unordered_map<std::string_view, int>;

  class MuonNumbering {
  public:
    const MuonBaseNumber geoHistoryToBaseNumber(const cms::ExpandedNodes&) const;
    const int get(const char*) const;
    void put(std::string_view, int);
    const MuonConstants& values() const { return values_; }

  private:
    MuonConstants values_;
  };
}  // namespace cms

#endif
