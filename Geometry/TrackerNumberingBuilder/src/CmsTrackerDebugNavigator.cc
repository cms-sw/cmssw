#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDebugNavigator.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDetExtra.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CmsTrackerDebugNavigator::CmsTrackerDebugNavigator ( const std::vector<GeometricDetExtra>& gdes ) {
  std::vector<GeometricDetExtra>::const_iterator gdeiEnd(gdes.end());
  std::vector<GeometricDetExtra>::const_iterator gdei(gdes.begin());
   for (; gdei != gdeiEnd; ++gdei) {
     const GeometricDetExtra* gdeaddr = &(*gdei);
     _helperMap[gdei->geographicalId().rawId()] = gdeaddr;
  }
}

void CmsTrackerDebugNavigator::dump(const GeometricDet& in, const std::vector<GeometricDetExtra>& gdes){
  edm::LogInfo("CmsTrackerDebugNavigator")<<" CmsTrackerDebugNavigator - Starting debug";
  for (int k=0; k<20; k++) numinstances[k]=0;
  iterate(in,0, gdes);
  
  for (int k=0; k<20; k++){
    edm::LogInfo("CmsTrackerDebugNavigator")<<" CmsTrackerDebugNavigator has found "<<numinstances[k]<<" GeometricDets at level "<<k;
  }
}

void CmsTrackerDebugNavigator::iterate(const GeometricDet& in, int level, const std::vector<GeometricDetExtra>& gdes){
  numinstances[level]++;
  for (unsigned int k=0; k<in.components().size(); k++){
    std::string spaces = "";
    for(unsigned int i=0; (int)i<=level; i++) spaces+="  ";
    const GeometricDetExtra* extra = _helperMap[in.components()[k]->geographicalId().rawId()];
    edm::LogInfo("CmsTrackerDebugNavigator") << level << spaces
					     << "### VOLUME " << in.components()[k]->name().name()
					     << " of type " << in.components()[k]->type()
					     << " copy number " << extra->copyno()
					     << " positioned in " << in.name().name()
					     << " global position of centre " << in.components()[k]->translation()
      //	      << " rotation matrix " << in.components()[k]->rotation()
					     << " volume = "  << extra->volume()  << " cm3"
					     << " density = " << extra->density() << " g/cm3"
					     << " weight "    << extra->weight()  << " g"
					     << std::endl;
    iterate(*((in.components())[k]),level+1, gdes);
  }
  return;
}
