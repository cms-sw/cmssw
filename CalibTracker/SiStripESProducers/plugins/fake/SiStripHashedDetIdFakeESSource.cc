#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripHashedDetIdFakeESSource.h"
#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
#include "CalibTracker/Records/interface/SiStripHashedDetIdRcd.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>
#include <vector>
#include <map>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripHashedDetIdFakeESSource::SiStripHashedDetIdFakeESSource( const edm::ParameterSet& pset )
  : SiStripHashedDetIdESProducer( pset )
{
  findingRecord<SiStripHashedDetIdRcd>();
  edm::LogVerbatim("HashedDetId") 
    << "[SiStripHashedDetIdFakeESSource::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripHashedDetIdFakeESSource::~SiStripHashedDetIdFakeESSource() {
  edm::LogVerbatim("HashedDetId")
    << "[SiStripHashedDetIdFakeESSource::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
SiStripHashedDetId* SiStripHashedDetIdFakeESSource::make( const SiStripHashedDetIdRcd& ) {
  edm::LogVerbatim("HashedDetId")
    << "[SiStripHashedDetIdFakeESSource::" << __func__ << "]"
    << " Building \"fake\" hashed DetId map from ascii file";
  
  typedef std::map<uint32_t,SiStripDetInfoFileReader::DetInfo>  Dets;
  const edm::Service<SiStripDetInfoFileReader> reader;
  Dets det_info = reader->getAllData();
  
  std::vector<uint32_t> dets;
  dets.reserve(16000);

  Dets::const_iterator idet = det_info.begin();
  Dets::const_iterator jdet = det_info.end();
  for ( ; idet != jdet; ++idet ) { dets.push_back( idet->first ); }
  edm::LogVerbatim("HashedDetId")
    << "[SiStripHashedDetIdESProducer::" << __func__ << "]"
    << " Retrieved " << dets.size()
    << " DetIds from ascii file!";
  
  SiStripHashedDetId* hash = new SiStripHashedDetId( dets );
  LogTrace("HashedDetId")
    << "[SiStripHashedDetIdESProducer::" << __func__ << "]"
    << " DetId hash map: " << std::endl
    << *hash;
  
  return hash;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripHashedDetIdFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& key, 
						     const edm::IOVSyncValue& iov_sync, 
						     edm::ValidityInterval& iov_validity ) {
  edm::ValidityInterval infinity( iov_sync.beginOfTime(), iov_sync.endOfTime() );
  iov_validity = infinity;
}
