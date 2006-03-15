#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDebugNavigator.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void CmsTrackerDebugNavigator::dump(const GeometricDet* in){
  edm::LogInfo("CmsTrackerDebugNavigator")<<" CmsTrackerDebugNavigator - Starting debug";
  for (int k=0; k<20; k++) numinstances[k]=0;
  iterate(in,0);
  
  for (int k=0; k<20; k++){
    edm::LogInfo("CmsTrackerDebugNavigator")<<" CmsTrackerDebugNavigator has found "<<numinstances[k]<<" GeometricDets at level "<<k;
  }
}


void CmsTrackerDebugNavigator::iterate(const GeometricDet*in, int level){
  static CmsTrackerStringToEnum enumName;
    
  numinstances[level]++;
  for (unsigned int k=0; k<(in)->components().size(); k++){
    iterate(((in)->components())[k],level+1);
  }
  return;
}
