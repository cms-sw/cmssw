#include "CalibTracker/SiStripESProducers/plugins/geom/SiStripHashedDetIdESModule.h"
#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
#include "CalibTracker/Records/interface/SiStripHashedDetIdRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripHashedDetIdESModule::SiStripHashedDetIdESModule( const edm::ParameterSet& pset ) 
  : SiStripHashedDetIdESProducer( pset )
{
  edm::LogVerbatim("HashedDetId") 
    << "[SiStripHashedDetIdESSourceFromGeom::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripHashedDetIdESModule::~SiStripHashedDetIdESModule() {
  edm::LogVerbatim("HashedDetId")
    << "[SiStripHashedDetIdESSourceFromGeom::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripHashedDetId* SiStripHashedDetIdESModule::make( const SiStripHashedDetIdRcd& rcd ) {
  edm::LogVerbatim("HashedDetId")
    << "[SiStripHashedDetIdFakeESSource::" << __func__ << "]"
    << " Building \"fake\" hashed DetId map from geometry";
  
  edm::ESHandle<TrackerGeometry> geom;
  rcd.getRecord<TrackerDigiGeometryRecord>().get( geom );
  
  std::vector<uint32_t> dets;
  dets.reserve(16000);

  TrackerGeometry::DetUnitContainer::const_iterator iter = geom->detUnits().begin();
  for( ; iter != geom->detUnits().end(); ++iter ) {
    const StripGeomDetUnit* strip = dynamic_cast<StripGeomDetUnit*>(*iter);
    if ( strip ) { dets.push_back( (strip->geographicalId()).rawId() ); }
  }
  edm::LogVerbatim(mlDqmCommon_)
    << "[SiStripHashedDetIdESModule::" << __func__ << "]"
    << " Retrieved " << dets.size()
    << " sistrip DetIds from geometry!";
  
  // Create hash map object
  SiStripHashedDetId* hash = new SiStripHashedDetId( dets );
  LogTrace(mlDqmCommon_)
    << "[SiStripHashedDetIdESModule::" << __func__ << "]"
    << " DetId hash map: " << std::endl
    << *hash;
  
  return hash;

}

