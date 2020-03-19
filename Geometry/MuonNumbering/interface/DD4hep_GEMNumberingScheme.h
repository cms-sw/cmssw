#ifndef GEOMETRY_MUON_NUMBERING_GEM_NUMBERING_SCHEME_H
#define GEOMETRY_MUON_NUMBERING_GEM_NUMBERING_SCHEME_H

/*
//\class GEMNumberingScheme

Description: GEM Numbering Scheme for DD4HEP

//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  Mon, 27 Jan 2020 
*/

#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"

class MuonBaseNumber;
class MuonConstants;

namespace cms {
  class GEMNumberingScheme {
  public:
    GEMNumberingScheme(const MuonConstants& muonConstants);
    void baseNumberToUnitNumber(const MuonBaseNumber&);
    int getDetId() const { return detId; }

  private:
    const int get(const char*, const MuonConstants&) const;
    void initMe(const MuonConstants& muonConstants);
    void setDetId(int idnew) { detId = idnew; }
    int theRegionLevel;
    int theStationLevel;
    int theRingLevel;
    int theSectorLevel;
    int theRollLevel;
    int detId;
  };
}  // namespace cms
#endif
