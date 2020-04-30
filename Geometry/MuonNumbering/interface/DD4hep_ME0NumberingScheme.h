#ifndef GEOMETRY_MUON_NUMBERING_ME0_NUMBERING_SCHEME_H
#define GEOMETRY_MUON_NUMBERING_ME0_NUMBERING_SCHEME_H

/*
//\class ME0NumberingScheme

Description: ME0 Numbering Scheme for DD4hep

//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osborne made for DTs (DD4HEP migration)
//          Created:  29 Apr 2020 
*/

#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"

class MuonBaseNumber;
class MuonConstants;

namespace cms {
  class ME0NumberingScheme {
  public:
    ME0NumberingScheme(const MuonConstants& muonConstants);
    void baseNumberToUnitNumber(const MuonBaseNumber&);
    int getDetId() const { return detId; }

  private:
    const int get(const char*, const MuonConstants&) const;
    void initMe(const MuonConstants& muonConstants);
    void setDetId(int idnew) { detId = idnew; }

    int theRegionLevel;
    int theSectorLevel;
    int theLayerLevel;
    int theRollLevel;
    int theNEtaPart;

    int detId;
  };
}  // namespace cms
#endif
