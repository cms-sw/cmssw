#ifndef PPSBASEDATA_H
#define PPSBASEDATA_H
#include <vector>
#include "FastSimulation/PPSFastObjects/interface/PPSTrackerHits.h"
#include "FastSimulation/PPSFastObjects/interface/PPSToFHits.h"
#include "TObject.h"

class PPSBaseData:public TObject {
public:
      PPSBaseData();
      virtual ~PPSBaseData() {};

      void AddHitTrk1(double x, double y) {TrkDet1.push_back(PPSTrackerHit(x,y));};
      void AddHitTrk2(double x, double y) {TrkDet2.push_back(PPSTrackerHit(x,y));};
      void AddHitToF(int cellid,double tof,double x, double y){ToFDet.push_back(PPSToFHit(cellid,tof,x,y));};

      virtual void clear() {TrkDet1.clear();TrkDet2.clear();ToFDet.clear();};

public:
      PPSTrackerHits      TrkDet1;
      PPSTrackerHits      TrkDet2;
      PPSToFHits          ToFDet;
ClassDef(PPSBaseData,1);
};
#endif
