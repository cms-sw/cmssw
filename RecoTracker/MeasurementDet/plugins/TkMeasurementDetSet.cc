#include "TkMeasurementDetSet.h"
#include "TkStripMeasurementDet.h"



void TkMeasurementDetSet::init() {
  // assume vector is full and ordered!
  int size = theStripDets.size();
  
  empty_.resize(size,true);
  activeThisEvent_.resize(size,true);
  activeThisPeriod_.resize(size,true);
  id_.resize(size);
  subId_.resize(size);
  totalStrips_.resize[size];
  
  bad128Strip_.resize(size*6);
  hasAny128StripBad_.resize(size);
  maskBad128StripBlocks_.resize(size);
  
  if (isRegional()) {
    detset_.resize(size);
  }  else {
    beginClusterI_.resize(size);
    endClusterI_.resize(size);
  }
  
  for (int i=0; i!=size; ++i) {
    auto & mdet =  *theStripDets[i]; 
    mdet.setIndex(i);
    //intialize the detId !
    id_[i] = mdet->gdet->geographicalId().rawId();
    subId_[i]=SiStripDetId(id_[i]).subdetId()-3;
    //initalize the total number of strips
    totalStrips_[i] =  mdet->specificGeomDet().specificTopology().nstrips();
  }
}
