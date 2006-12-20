/** \file GlobalTrackingGeometry.cc
 *
 *  $Date: 2006/07/12 11:00:59 $
 *  $Revision: 1.6 $
 *  \author M. Sani
 */

#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>

using namespace edm;

GlobalTrackingGeometry::GlobalTrackingGeometry(std::vector<const TrackingGeometry*>& geos) : theGeometries(geos) {}

GlobalTrackingGeometry::~GlobalTrackingGeometry() {}

const GeomDetUnit* GlobalTrackingGeometry::idToDetUnit(DetId id) const {
    
    const TrackingGeometry* tg = slaveGeometry(id);
    
    if (tg != 0) {
      return tg->idToDetUnit(id);
    } else {
      return 0;
    }
}


const GeomDet* GlobalTrackingGeometry::idToDet(DetId id) const{
  
    const TrackingGeometry* tg = slaveGeometry(id);
    
    if (tg != 0) {
        return tg->idToDet(id);
    } else {
      return 0;
    }
}

const TrackingGeometry* GlobalTrackingGeometry::slaveGeometry(DetId id) const {  
  
    int idx = id.det()-1;
    if (id.det() == DetId::Muon) {
        
        idx+=id.subdetId()-1;
    }

    if (theGeometries[idx]==0) throw cms::Exception("NoGeometry") << "No Tracking Geometry is available for DetId " << id.rawId() << std::endl;

    return theGeometries[idx];
}

const TrackingGeometry::DetTypeContainer& GlobalTrackingGeometry::detTypes() const {
    
   static DetTypeContainer result;
   if ( ! result.empty() ) return result; 
   for(std::vector<const TrackingGeometry*>::const_iterator geom = theGeometries.begin();
       geom != theGeometries.end(); geom++)
     {
	if (*geom == 0) continue;
	DetTypeContainer detTypes((*geom)->detTypes());
	if ( detTypes.size()+result.size()<result.capacity() ) result.resize(detTypes.size()+result.size());
	for( DetTypeContainer::const_iterator detType = detTypes.begin(); detType!=detTypes.end(); detType++)
	  result.push_back(*detType);
     }
   return result;
}

const TrackingGeometry::DetUnitContainer& GlobalTrackingGeometry::detUnits() const {

   static DetUnitContainer result;
   if ( ! result.empty() ) return result; 
   for(std::vector<const TrackingGeometry*>::const_iterator geom = theGeometries.begin();
       geom != theGeometries.end(); geom++)
     {
	if (*geom == 0) continue;
	DetUnitContainer detUnits((*geom)->detUnits());
	if ( detUnits.size()+result.size()<result.capacity() ) result.resize(detUnits.size()+result.size());
	for( DetUnitContainer::const_iterator detUnit = detUnits.begin(); detUnit!=detUnits.end(); detUnit++)
	  result.push_back(*detUnit);
     }
   return result;
}

const TrackingGeometry::DetContainer& GlobalTrackingGeometry::dets() const {

   static DetContainer result;
   if ( ! result.empty() ) return result; 
   for(std::vector<const TrackingGeometry*>::const_iterator geom = theGeometries.begin();
       geom != theGeometries.end(); geom++)
     {
	if (*geom == 0) continue;
	DetContainer dets((*geom)->dets());
	if ( dets.size()+result.size()<result.capacity() ) result.resize(dets.size()+result.size());
	for( DetContainer::const_iterator det = dets.begin(); det!=dets.end(); det++)
	  result.push_back(*det);
     }
   return result;
}

const TrackingGeometry::DetIdContainer& GlobalTrackingGeometry::detUnitIds() const {

   static DetIdContainer result;
   if ( ! result.empty() ) return result; 
   for(std::vector<const TrackingGeometry*>::const_iterator geom = theGeometries.begin();
       geom != theGeometries.end(); geom++)
     {
	if (*geom == 0) continue;
	DetIdContainer detUnitIds((*geom)->detUnitIds());
	if ( detUnitIds.size()+result.size()<result.capacity() ) result.resize(detUnitIds.size()+result.size());
	for( DetIdContainer::const_iterator detUnitId = detUnitIds.begin(); detUnitId!=detUnitIds.end(); detUnitId++)
	  result.push_back(*detUnitId);
     }
   return result;
}

const TrackingGeometry::DetIdContainer& GlobalTrackingGeometry::detIds() const {

   static DetIdContainer result;
   if ( ! result.empty() ) return result; 
   for(std::vector<const TrackingGeometry*>::const_iterator geom = theGeometries.begin();
       geom != theGeometries.end(); geom++)
     {
	if (*geom == 0) continue;
	DetIdContainer detIds((*geom)->detIds());
	if ( detIds.size()+result.size()<result.capacity() ) result.resize(detIds.size()+result.size());
	for( DetIdContainer::const_iterator detId = detIds.begin(); detId!=detIds.end(); detId++)
	  result.push_back(*detId);
     }
   return result;
}
