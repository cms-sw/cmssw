#include "Geometry/TrackerNumberingBuilder/interface/TrackerMapDDDtoID.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<algorithm>

TrackerMapDDDtoID::TrackerMapDDDtoID(const GeometricDet* iDet) 
{
   buildAll(iDet);
  //
  // loop over all the volumes which hahe a SpecPar called specpar with value value, 
  // and save in the map the association nav_type<->id
  //
}

void TrackerMapDDDtoID::buildAll(const GeometricDet* iDet){
  edm::LogInfo("TrackerMapDDDtoID")<<" Building the TrackerMapDDDtoID map.";
  TrackerMapDDDtoID* me = const_cast<TrackerMapDDDtoID*>(this);
  me->buildAllStep2(iDet);
}

void TrackerMapDDDtoID::buildAllStep2(const GeometricDet* theTracker){
  
  std::vector<const GeometricDet*> allDetectors;
  theTracker->deepComponents(allDetectors);
  
  //
  // Also build a map! (for slower access)
  //

  for (auto & allDetector : allDetectors){
    
    path2id_.insert(std::pair<nav_type,uint32_t>(allDetector->navType(),(allDetector->geographicalID())()));
    revpath2id_.insert(std::pair<uint32_t,nav_type>((allDetector->geographicalID())(),allDetector->navType()));
    navVec.emplace_back(allDetector->navType());
  }
  edm::LogInfo("TrackerMapDDDtoID")<<"Created TrackerMapDDDtoID; results in "<<allDetectors.size()<<" detectors numbered.";
}

/*
unsigned int TrackerMapDDDtoID::id(const DDExpandedView & e) const
{
  return id(e.navPos());
}


unsigned int TrackerMapDDDtoID::id(const DDFilteredView & f) const
{
  return id(f.navPos());
}
*/

unsigned int TrackerMapDDDtoID::id(const nav_type & n) const
{
  std::map<nav_type,uint32_t>::const_iterator it = path2id_.find(n);
  unsigned int result = 0;
  if (it != path2id_.end())
    result = it->second;
  return result;  
}


std::vector<TrackerMapDDDtoID::nav_type> const & TrackerMapDDDtoID::allNavTypes() const{
  return navVec;
}

namespace {
  const TrackerMapDDDtoID::nav_type nullresult;
}

TrackerMapDDDtoID::nav_type const & TrackerMapDDDtoID::navType(uint32_t num) const
{ 
  std::map<uint32_t,nav_type>::const_iterator it = revpath2id_.find(num);
  if (it != revpath2id_.end())
    return it->second;
  return nullresult;  
}

void TrackerMapDDDtoID::clear(){
  path2id_.clear();
  edm::LogInfo("TrackerMapDDDtoID")<<" TrackerMapDDDtoID maps deleted from memory.";
}
