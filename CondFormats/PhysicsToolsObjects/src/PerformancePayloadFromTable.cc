#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTable.h"

int PerformancePayloadFromTable::InvalidPos=-1;

#include <iostream>


float PerformancePayloadFromTable::getResult(PerformanceResult::ResultType r ,BinningPointByMap p) const {

  if (! isInPayload(r,p)) return  PerformancePayload::InvalidResult;

  // loop on the table rows and search for a match
  for (int i=0; i< pl.nRows(); i++){
    PhysicsPerformancePayload::Row  row = pl.getRow(i);

    if (matches(p,row)){
      int pos = resultPos(r);
      return row[pos];
    }
  }
  return PerformancePayload::InvalidResult;
}

bool PerformancePayloadFromTable::matches(BinningPointByMap p, PhysicsPerformancePayload::Row & row) const {
  //
  // this is the smart function which does not take into account the fields not present 
  //

  // I can do it via a loop!
  
    std::vector<BinningVariables::BinningVariablesType> t = myBinning();


    for (std::vector<BinningVariables::BinningVariablesType>::const_iterator it = t.begin(); it != t.end();++it){
      //
      // first the binning point map must contain ALL the quantities here
      //
      //    if (! p.isKeyAvailable(*it) ) return false;
      float v = p.value(*it);
      if (!(v >= row[minPos(*it)] && v  < row[maxPos(*it)])) return false;
  }
  return true;
}

bool PerformancePayloadFromTable::isInPayload(PerformanceResult::ResultType res,BinningPointByMap point) const {
  // first, let's see if it is available at all
  if (resultPos(res) == PerformancePayloadFromTable::InvalidPos) return false;
  // now look whther the binning point contains all the info
  std::vector<BinningVariables::BinningVariablesType> t = myBinning();
  for (std::vector<BinningVariables::BinningVariablesType>::const_iterator it = t.begin(); it != t.end();++it){
    if (! point.isKeyAvailable(*it) ) return false;
  }
  // then, look if there is a matching row
  for (int i=0; i< pl.nRows(); i++){
    PhysicsPerformancePayload::Row  row = pl.getRow(i);
    if (matches(point,row)){
      return true;
    }
  }
  return false;
}

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(PerformancePayloadFromTable);
