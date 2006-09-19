#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDebugNavigator.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
    std::string spaces = "";
    for(unsigned int i=0; (int)i<=level; i++) spaces+="  ";
    edm::LogInfo("CmsTrackerDebugNavigator") << level << spaces
					     << "### VOLUME " << (in)->components()[k]->name().name()
					     << " copy number " << (in)->components()[k]->copyno()
					     << " positioned in " << (in)->name().name()
					     << " global position of centre " << (in)->components()[k]->translation()
      //	      << " rotation matrix " << (in)->components()[k]->rotation()
					     << " volume = "  << (in)->components()[k]->volume()  << " cm3"
					     << " density = " << (in)->components()[k]->density() << " g/cm3"
					     << " weight "    << (in)->components()[k]->weight()  << " g"
					     << std::endl;
    iterate(((in)->components())[k],level+1);
  }
  return;
}
