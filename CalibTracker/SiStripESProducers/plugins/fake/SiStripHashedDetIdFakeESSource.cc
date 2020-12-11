#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/utils.h"

#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
#include "CalibTracker/Records/interface/SiStripHashedDetIdRcd.h"

class SiStripHashedDetIdFakeESSource : public edm::ESProducer {
public:
  explicit SiStripHashedDetIdFakeESSource(const edm::ParameterSet&);
  ~SiStripHashedDetIdFakeESSource() override;

  virtual std::unique_ptr<SiStripHashedDetId> produce(const SiStripHashedDetIdRcd&);

private:
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
};

using namespace sistrip;

SiStripHashedDetIdFakeESSource::SiStripHashedDetIdFakeESSource(const edm::ParameterSet& pset)
    : geomDetToken_(setWhatProduced(this).consumes()) {}

SiStripHashedDetIdFakeESSource::~SiStripHashedDetIdFakeESSource() {}

std::unique_ptr<SiStripHashedDetId> SiStripHashedDetIdFakeESSource::produce(const SiStripHashedDetIdRcd& record) {
  edm::LogVerbatim("HashedDetId") << "[SiStripHashedDetIdFakeESSource::" << __func__ << "]"
                                  << " Building \"fake\" hashed DetId map from IdealGeometry";

  const auto& geomDet = record.getRecord<TrackerDigiGeometryRecord>().get(geomDetToken_);

  const std::vector<uint32_t> dets = TrackerGeometryUtils::getSiStripDetIds(geomDet);
  edm::LogVerbatim("HashedDetId") << "[SiStripHashedDetIdFakeESSource::" << __func__ << "]"
                                  << " Retrieved " << dets.size() << " DetIds from IdealGeometry!";

  auto hash = std::make_unique<SiStripHashedDetId>(dets);
  LogTrace("HashedDetId") << "[SiStripHashedDetIdFakeESSource::" << __func__ << "]"
                          << " DetId hash map: " << std::endl
                          << *hash;

  return hash;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(SiStripHashedDetIdFakeESSource);
