#include "CalibTracker/SiStripESProducers/plugins/stubs/SiStripHashedDetIdESProducer.h"
#include "CalibTracker/Records/interface/SiStripHashedDetIdRcd.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripHashedDetIdESProducer::SiStripHashedDetIdESProducer( const edm::ParameterSet& pset ) {
  setWhatProduced( this, &SiStripHashedDetIdESProducer::produce );
}

// -----------------------------------------------------------------------------
//
SiStripHashedDetIdESProducer::~SiStripHashedDetIdESProducer() {}

// -----------------------------------------------------------------------------
//
std::auto_ptr<SiStripHashedDetId> SiStripHashedDetIdESProducer::produce( const SiStripHashedDetIdRcd& rcd ) {

  // Retrieve geometry
  edm::ESHandle<TrackerGeometry> geom;
  rcd.getRecord<TrackerDigiGeometryRecord>().get( geom );
  
  // Build list of DetIds
  std::vector<uint32_t> dets;
  dets.reserve(16000);
  TrackerGeometry::DetUnitContainer::const_iterator iter = geom->detUnits().begin();
  for( ; iter != geom->detUnits().end(); ++iter ) {
    const StripGeomDetUnit* strip = dynamic_cast<StripGeomDetUnit*>(*iter);
    if( strip ) {
      dets.push_back( (strip->geographicalId()).rawId() );
    }
  }
  edm::LogVerbatim(mlDqmCommon_)
    << "[SiStripHashedDetIdESProducer::" << __func__ << "]"
    << " Retrieved " << dets.size()
    << " sistrip DetIds from geometry!";
  
  // Create hash map object
  SiStripHashedDetId* hash = new SiStripHashedDetId( dets );
  LogTrace(mlDqmCommon_)
    << "[SiStripHashedDetIdESProducer::" << __func__ << "]"
    << " DetId hash map: " << std::endl
    << *hash;
  
  return std::auto_ptr<SiStripHashedDetId>( hash );

}

