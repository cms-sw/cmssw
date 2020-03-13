#ifndef Geometry_MuonNumbering_cms_CSCNumberingScheme_h
#define Geometry_MuonNumbering_cms_CSCNumberingScheme_h
/*
// \class CSCNumberingScheme
//
//  Description: CSC Numbering Scheme for DD4hep
//              
//
// \author Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//         Created:  Thu, 05 March 2020 
//   
//         Old DD version authors:  Arno Straessner & Tim Cox
*/
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"

class MuonBaseNumber;
class MuonConstants;

namespace cms {
  class CSCNumberingScheme {
  public:
    CSCNumberingScheme(const MuonConstants& muonConstants);
    void baseNumberToUnitNumber(const MuonBaseNumber&);
    int getDetId() const { return detId; }

  private:
    const int get(const char*, const MuonConstants&) const;
    void initMe(const MuonConstants& muonConstants);
    void setDetId(int idnew) { detId = idnew; }

    int chamberIndex(int, int, int, int) const;

    int theRegionLevel;
    int theStationLevel;
    int theSubringLevel;
    int theSectorLevel;
    int theLayerLevel;
    int theRingLevel;

    int detId;
  };
}  // namespace cms
#endif
