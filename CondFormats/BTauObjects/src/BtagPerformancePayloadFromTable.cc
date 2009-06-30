#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTable.h"

int BtagPerformancePayloadFromTable::InvalidPos=-1;

#include <iostream>


float BtagPerformancePayloadFromTable::getResult(BtagResult::BtagResultType r ,BtagBinningPointByMap p) const {
  //
  // dump the table here
  //

  /*  
  for (int k=0;k<(table_.size()/stride_); k++){
    for (int j=0; j<stride_; j++){
      std::cout << "PPos["<<k<<","<<j<<"] = "<<table_[k*stride_+j]<<std::endl;
    }
  }
  */


  // loop on the table rows and search for a match
  for (int i=0; i< nRows(); i++){
    PhysicsPerformancePayload::Row  row = getRow(i);

    if (matches(p,row)){
      int pos = resultPos(r);
      return row[pos];
    }
  }
  return BtagPerformancePayload::InvalidResult;
}

bool BtagPerformancePayloadFromTable::matches(BtagBinningPointByMap p, PhysicsPerformancePayload::Row & row) const {
  //
  // this is the smart function which does not take into account the fields not present 
  //

  // I can do it via a loop!
  
    std::vector<BtagBinningPointByMap::BtagBinningPointType> t = myBinning();


  for (std::vector<BtagBinningPointByMap::BtagBinningPointType>::const_iterator it = t.begin(); it != t.end();++it){
    //
    // first the binning point map must contain ALL the quantities here
    //
    //    if (! p.isKeyAvailable(*it) ) return false;
    float v = p.value(*it);
    if (!(v > row[minPos(*it)] && v  < row[maxPos(*it)])) return false;
  }
  return true;

}

bool BtagPerformancePayloadFromTable::isInPayload(BtagResult::BtagResultType res,BtagBinningPointByMap point) const {
  // first, let's see if it is available at all
  if (resultPos(res) == BtagPerformancePayloadFromTable::InvalidPos) return false;
  // now look whther the binning point contains all the info
  std::vector<BtagBinningPointByMap::BtagBinningPointType> t = myBinning();
  for (std::vector<BtagBinningPointByMap::BtagBinningPointType>::const_iterator it = t.begin(); it != t.end();++it){
    if (! point.isKeyAvailable(*it) ) return false;
  }
  // then, look if there is a matching row
  for (int i=0; i< nRows(); i++){
    PhysicsPerformancePayload::Row  row = getRow(i);
    if (matches(point,row)){
      return true;
    }
  }
  return false;
}

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
EVENTSETUP_DATA_REG(BtagPerformancePayloadFromTable);
