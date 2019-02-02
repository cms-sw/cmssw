#ifndef GEOMETRY_RECO_GEOMETRY_DT_NUMBERING_SCHEME_H
#define GEOMETRY_RECO_GEOMETRY_DT_NUMBERING_SCHEME_H

#include "DetectorDescription/DDCMS/interface/MuonNumbering.h"

namespace cms {
  struct DTNumberingScheme {
    DTNumberingScheme(MuonConstants&);
    int baseNumberToUnitNumber(const MuonBaseNumber&);
    int getDetId(const MuonBaseNumber&) const;

    void initMe(MuonConstants&);
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
