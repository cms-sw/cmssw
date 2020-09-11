#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
#include "CalibTracker/Records/interface/SiStripHashedDetIdRcd.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

using namespace sistrip;

/**
   @class SiStripHashedDetIdESModule
   @author R.Bainbridge
   @brief Builds hashed DetId map based on DetIds read from geometry database
*/
class SiStripHashedDetIdESModule : public edm::ESProducer {
public:
  SiStripHashedDetIdESModule(const edm::ParameterSet&);
  ~SiStripHashedDetIdESModule() override;

  /** Builds hashed DetId map based on geometry. */
  std::unique_ptr<SiStripHashedDetId> produce(const SiStripHashedDetIdRcd&);

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
};

// -----------------------------------------------------------------------------
//
SiStripHashedDetIdESModule::SiStripHashedDetIdESModule(const edm::ParameterSet& pset)
    : geomToken_(setWhatProduced(this, &SiStripHashedDetIdESModule::produce).consumes()) {
  edm::LogVerbatim("HashedDetId") << "[SiStripHashedDetIdESSourceFromGeom::" << __func__ << "]"
                                  << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripHashedDetIdESModule::~SiStripHashedDetIdESModule() {
  edm::LogVerbatim("HashedDetId") << "[SiStripHashedDetIdESSourceFromGeom::" << __func__ << "]"
                                  << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
std::unique_ptr<SiStripHashedDetId> SiStripHashedDetIdESModule::produce(const SiStripHashedDetIdRcd& rcd) {
  edm::LogVerbatim("HashedDetId") << "[SiStripHashedDetIdFakeESSource::" << __func__ << "]"
                                  << " Building \"fake\" hashed DetId map from geometry";

  const auto& geom = rcd.get(geomToken_);

  std::vector<uint32_t> dets;
  dets.reserve(16000);

  for (const auto& iter : geom.detUnits()) {
    const auto strip = dynamic_cast<StripGeomDetUnit const*>(iter);
    if (strip) {
      dets.push_back((strip->geographicalId()).rawId());
    }
  }
  edm::LogVerbatim(mlDqmCommon_) << "[SiStripHashedDetIdESModule::" << __func__ << "]"
                                 << " Retrieved " << dets.size() << " sistrip DetIds from geometry!";

  // Create hash map object
  auto hash = std::make_unique<SiStripHashedDetId>(dets);
  LogTrace(mlDqmCommon_) << "[SiStripHashedDetIdESModule::" << __func__ << "]"
                         << " DetId hash map: " << std::endl
                         << *hash;

  return hash;
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripHashedDetIdESModule);
