#ifndef GEOMETRY_MUON_NUMBERING_RPC_NUMBERING_SCHEME_H
#define GEOMETRY_MUON_NUMBERING_RPC_NUMBERING_SCHEME_H

/*
//\class RPCNumberingScheme

 Description: RPC Numbering Scheme for DD4hep
              
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  Fri, 20 Sep 2019 
*/

#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"

class MuonBaseNumber;
class MuonConstants;

namespace cms {
  class RPCNumberingScheme {
  public:
    RPCNumberingScheme(const MuonConstants& muonConstants);
    void baseNumberToUnitNumber(const MuonBaseNumber&);
    int getDetId() const { return detId; }

  private:
    const int get(const char*, const MuonConstants&) const;
    void initMe(const MuonConstants& muonConstants);
    void setDetId(int idnew) { detId = idnew; }
    int theRegionLevel;
    int theBWheelLevel;
    int theBStationLevel;
    int theBPlaneLevel;
    int theBChamberLevel;
    int theEPlaneLevel;
    int theESectorLevel;
    int theERollLevel;

    int detId;
  };
}  // namespace cms
#endif
