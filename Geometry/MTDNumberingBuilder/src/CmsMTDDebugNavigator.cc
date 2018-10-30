#include "Geometry/MTDNumberingBuilder/interface/CmsMTDDebugNavigator.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDetExtra.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CmsMTDDebugNavigator::CmsMTDDebugNavigator ( const std::vector<GeometricTimingDetExtra>& gdes ) {
  for( const auto& gdei : gdes ) {
    const GeometricTimingDetExtra* gdeaddr = &gdei;
    _helperMap[gdei.geographicalId().rawId()] = gdeaddr;
  }
}

void CmsMTDDebugNavigator::dump(const GeometricTimingDet& in, const std::vector<GeometricTimingDetExtra>& gdes){
  edm::LogInfo("CmsMTDDebugNavigator")<<" CmsMTDDebugNavigator - Starting debug";
  for (int k=0; k<20; k++) numinstances[k]=0;
  iterate(in,0, gdes);
  
  for (int k=0; k<20; k++){
    edm::LogInfo("CmsMTDDebugNavigator")<<" CmsMTDDebugNavigator has found "<<numinstances[k]<<" GeometricTimingDets at level "<<k;
  }
}

void CmsMTDDebugNavigator::iterate(const GeometricTimingDet& in, int level, const std::vector<GeometricTimingDetExtra>& gdes){
  numinstances[level]++;
  std::string spaces(' ',level*2);
  for (unsigned int k=0; k<in.components().size(); k++){    
    const GeometricTimingDetExtra* extra = _helperMap[in.components()[k]->geographicalId().rawId()];
    edm::LogInfo("CmsMTDDebugNavigator") << level << spaces
					     << "### VOLUME " << in.components()[k]->name().name()
					     << " of type " << in.components()[k]->type()
					     << " copy number " << extra->copyno()
					     << " positioned in " << in.name().name()
					     << " global position of centre " << in.components()[k]->translation()
					     << " volume = "  << extra->volume()  << " cm3"
					     << " density = " << extra->density() << " g/cm3"
					     << " weight "    << extra->weight()  << " g"
					     << std::endl;
    iterate(*((in.components())[k]),level+1, gdes);
  }
  return;
}
