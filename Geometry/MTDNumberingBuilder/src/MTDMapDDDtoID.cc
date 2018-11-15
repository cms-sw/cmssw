#include "Geometry/MTDNumberingBuilder/interface/MTDMapDDDtoID.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<algorithm>

MTDMapDDDtoID::MTDMapDDDtoID(const GeometricTimingDet* iDet) 
{
   buildAll(iDet);
  //
  // loop over all the volumes which hahe a SpecPar called specpar with value value, 
  // and save in the map the association nav_type<->id
  //
}

void MTDMapDDDtoID::buildAll(const GeometricTimingDet* iDet){
  edm::LogInfo("MTDMapDDDtoID")<<" Building the MTDMapDDDtoID map.";
  MTDMapDDDtoID* me = const_cast<MTDMapDDDtoID*>(this);
  me->buildAllStep2(iDet);
}

void MTDMapDDDtoID::buildAllStep2(const GeometricTimingDet* theTracker){
  
  std::vector<const GeometricTimingDet*> allDetectors;
  theTracker->deepComponents(allDetectors);
  
  //
  // Also build a map! (for slower access)
  //

  for (auto & allDetector : allDetectors){
    
    path2id_.insert(std::pair<nav_type,uint32_t>(allDetector->navType(),(allDetector->geographicalID())()));
    revpath2id_.insert(std::pair<uint32_t,nav_type>((allDetector->geographicalID())(),allDetector->navType()));
    navVec.emplace_back(allDetector->navType());
  }
  edm::LogInfo("MTDMapDDDtoID")<<"Created MTDMapDDDtoID; results in "<<allDetectors.size()<<" detectors numbered.";
}

unsigned int MTDMapDDDtoID::id(const nav_type & n) const
{
  std::map<nav_type,uint32_t>::const_iterator it = path2id_.find(n);
  unsigned int result = 0;
  if (it != path2id_.end())
    result = it->second;
  return result;  
}


std::vector<MTDMapDDDtoID::nav_type> const & MTDMapDDDtoID::allNavTypes() const{
  return navVec;
}

namespace {
  const MTDMapDDDtoID::nav_type nullresult;
}

MTDMapDDDtoID::nav_type const & MTDMapDDDtoID::navType(uint32_t num) const
{ 
  std::map<uint32_t,nav_type>::const_iterator it = revpath2id_.find(num);
  if (it != revpath2id_.end())
    return it->second;
  return nullresult;  
}

void MTDMapDDDtoID::clear(){
  path2id_.clear();
  edm::LogInfo("MTDMapDDDtoID")<<" MTDMapDDDtoID maps deleted from memory.";
}
