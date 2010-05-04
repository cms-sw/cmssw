#ifndef RecoMET_MuonMETInfo_h
#define RecoMET_MuonMETInfo_h

#include "DataFormats/Math/interface/Point3D.h"

struct MuonMETInfo {

  float ecalE;
  float hcalE;
  float hoE;
  math::XYZPoint ecalPos;
  math::XYZPoint hcalPos;
  math::XYZPoint hoPos;
  bool useAverage;
  //if using in FWLite, this should be false
  bool useTkAssociatorPositions;
  bool useHO;

  MuonMETInfo():
  ecalE(0), hcalE(0), hoE(0),
  ecalPos(0,0,0),hcalPos(0,0,0), hoPos(0,0,0),
  useAverage(0), useTkAssociatorPositions(0),useHO(0){ }

};
  

#endif








