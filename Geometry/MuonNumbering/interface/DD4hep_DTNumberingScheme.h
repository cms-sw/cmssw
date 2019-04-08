#ifndef GEOMETRY_MUON_NUMBERING_DT_NUMBERING_SCHEME_H
#define GEOMETRY_MUON_NUMBERING_DT_NUMBERING_SCHEME_H

// -*- C++ -*-
//
// Package:    Geometry/MuonNumbering
// Class:      DTNumberingScheme
// 
/**\class DTNumberingScheme

 Description: DTNumberingScheme converts the MuonBaseNumber
 to a unit id for Muon Barrel

 Implementation:
     DTNumberingScheme decode and getDetId are ported from
     an original DTNumberingScheme class
*/
//
// Original Author:  Ianna Osborne
//         Created:  Thu, 21 Mar 2019 15:18:08 CET
//
//

#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"

namespace cms {
  class DTNumberingScheme {
  public:
    DTNumberingScheme(const MuonConstants&);
    int baseNumberToUnitNumber(const MuonBaseNumber&);
    int getDetId(const MuonBaseNumber&) const;

  private:
    void initMe(const MuonConstants&);
    const int get(const char*, const MuonConstants&) const;
    // Decode MuonBaseNumber to id: no checking
    void decode(const MuonBaseNumber& num,
		int& wire_id,
		int& layer_id,
		int& superlayer_id,
		int& sector_id,
		int& station_id,
		int& wheel_id
		) const;

    int theRegionLevel;
    int theWheelLevel;
    int theStationLevel;
    int theSuperLayerLevel;
    int theLayerLevel;
    int theWireLevel;
  };
}

#endif
