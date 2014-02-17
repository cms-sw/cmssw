/** \file GlobalTrackingGeometry.cc
 *
 *  $Date: 2012/07/24 13:48:29 $
 *  $Revision: 1.8 $
 *  \author M. Sani
 */

#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>
#include <FWCore/Utilities/interface/Exception.h>

GlobalTrackingGeometry::GlobalTrackingGeometry(std::vector<const TrackingGeometry*>& geos)
    : theGeometries(geos)
{}

GlobalTrackingGeometry::~GlobalTrackingGeometry()
{}

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

const TrackingGeometry::DetTypeContainer&
GlobalTrackingGeometry::detTypes( void ) const
{    
   if ( ! theDetTypes.empty() ) return theDetTypes; 
   for( std::vector<const TrackingGeometry*>::const_iterator geom = theGeometries.begin(), geomEnd = theGeometries.end();
       geom != geomEnd; ++geom )
     {
	if( *geom == 0 ) continue;
	DetTypeContainer detTypes(( *geom )->detTypes());
	if( detTypes.size() + theDetTypes.size() < theDetTypes.capacity()) theDetTypes.resize( detTypes.size() + theDetTypes.size());
	for( DetTypeContainer::const_iterator detType = detTypes.begin(), detTypeEnd = detTypes.end(); detType != detTypeEnd; ++detType )
	  theDetTypes.push_back( *detType );
     }
   return theDetTypes;
}

const TrackingGeometry::DetUnitContainer&
GlobalTrackingGeometry::detUnits( void ) const
{
   if( ! theDetUnits.empty()) return theDetUnits; 
   for( std::vector<const TrackingGeometry*>::const_iterator geom = theGeometries.begin(), geomEnd = theGeometries.end();
       geom != geomEnd; ++geom )
     {
	if( *geom == 0 ) continue;
	DetUnitContainer detUnits(( *geom )->detUnits());
	if( detUnits.size() + theDetUnits.size() < theDetUnits.capacity()) theDetUnits.resize( detUnits.size() + theDetUnits.size());
	for( DetUnitContainer::const_iterator detUnit = detUnits.begin(), detUnitEnd = detUnits.end(); detUnit != detUnitEnd; ++detUnit )
	  theDetUnits.push_back( *detUnit );
     }
   return theDetUnits;
}

const TrackingGeometry::DetContainer&
GlobalTrackingGeometry::dets( void ) const
{
   if( ! theDets.empty()) return theDets; 
   for( std::vector<const TrackingGeometry*>::const_iterator geom = theGeometries.begin(), geomEnd = theGeometries.end();
       geom != geomEnd; ++geom )
     {
	if( *geom == 0 ) continue;
	DetContainer dets(( *geom )->dets());
	if( dets.size() + theDets.size() < theDets.capacity()) theDets.resize( dets.size() + theDets.size());
	for( DetContainer::const_iterator det = dets.begin(), detEnd = dets.end(); det != detEnd; ++det )
	  theDets.push_back( *det );
     }
   return theDets;
}

const TrackingGeometry::DetIdContainer&
GlobalTrackingGeometry::detUnitIds( void ) const
{
   if( ! theDetUnitIds.empty()) return theDetUnitIds; 
   for( std::vector<const TrackingGeometry*>::const_iterator geom = theGeometries.begin(), geomEnd = theGeometries.end();
       geom != geomEnd; ++geom )
     {
	if( *geom == 0 ) continue;
	DetIdContainer detUnitIds(( *geom )->detUnitIds());
	if( detUnitIds.size() + theDetUnitIds.size() < theDetUnitIds.capacity()) theDetUnitIds.resize( detUnitIds.size() + theDetUnitIds.size());
	for( DetIdContainer::const_iterator detUnitId = detUnitIds.begin(), detUnitIdEnd = detUnitIds.end(); detUnitId != detUnitIdEnd; ++detUnitId )
	  theDetUnitIds.push_back( *detUnitId );
     }
   return theDetUnitIds;
}

const TrackingGeometry::DetIdContainer&
GlobalTrackingGeometry::detIds( void ) const
{
   if( ! theDetIds.empty() ) return theDetIds; 
   for( std::vector<const TrackingGeometry*>::const_iterator geom = theGeometries.begin(), geomEnd = theGeometries.end();
       geom != geomEnd; ++geom )
     {
	if( *geom == 0 ) continue;
	DetIdContainer detIds(( *geom )->detIds());
	if( detIds.size() + theDetIds.size() < theDetIds.capacity()) theDetIds.resize( detIds.size() + theDetIds.size());
	for( DetIdContainer::const_iterator detId = detIds.begin(), detIdEnd = detIds.end(); detId != detIdEnd; ++detId )
	  theDetIds.push_back( *detId );
     }
   return theDetIds;
}
