#ifndef GEOMETRY_MUON_NUMBERING_RPC_NUMBERING_SCHEME_H
#define GEOMETRY_MUON_NUMBERING_RPC_NUMBERING_SCHEME_H

// -*- C++ -*-
//
//
/*

 Description: RPC Numbering Scheme for DD4HEP 
              based on DT Numbering Scheme made by Ianna Osburne 

*/
//
//         Author:  Sergio Lo Meo (INFN Section of Bologna - Italy) sergio.lomeo@cern.ch
//         Created:  Wed, 21 August 2019 16:00 CET
//
//

#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"

class MuonBaseNumber;
class MuonConstants;

namespace cms {
class RPCNumberingScheme {
public:

  RPCNumberingScheme(const MuonConstants& muonConstants);

  void baseNumberToUnitNumber(const MuonBaseNumber&);
  void SetDetId(int idnew){detId=idnew;}
  int GetDetId()const {return detId;}

private:

  const int get(const char*, const MuonConstants&) const;
  void initMe(const MuonConstants& muonConstants);

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
