// system includes
#include <sys/time.h>
#include <memory>

// CLHEP
#include "CLHEP/Random/RandGauss.h"

// user includes
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

namespace cms {
  class SiPixelPerformanceSummaryBuilder : public edm::one::EDAnalyzer<> {
  public:
    explicit SiPixelPerformanceSummaryBuilder(const edm::ParameterSet&);
    ~SiPixelPerformanceSummaryBuilder() override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
    std::vector<uint32_t> detectorModules_;
  };
}  // namespace cms

using namespace cms;

SiPixelPerformanceSummaryBuilder::SiPixelPerformanceSummaryBuilder(const edm::ParameterSet& iConfig)
    : tkGeomToken_(esConsumes()) {}

SiPixelPerformanceSummaryBuilder::~SiPixelPerformanceSummaryBuilder() = default;

void SiPixelPerformanceSummaryBuilder::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const TrackerGeometry* pDD = &iSetup.getData(tkGeomToken_);
  edm::LogInfo("SiPixelPerformanceSummaryBuilder") << pDD->detUnits().size() << " detectors" << std::endl;

  for (const auto& it : pDD->detUnits()) {
    if (dynamic_cast<PixelGeomDetUnit const*>(it) != nullptr) {
      detectorModules_.push_back(it->geographicalId().rawId());
    }
  }
  edm::LogInfo("Modules") << "detectorModules_.size() = " << detectorModules_.size();

  SiPixelPerformanceSummary performanceSummary;

  for (std::vector<uint32_t>::const_iterator iDet = detectorModules_.begin();  // fill object
       iDet != detectorModules_.end();
       ++iDet) {
    float nDigisMean = (float)CLHEP::RandGauss::shoot(50., 20.);  // generate random values for each detId
    float nDigisRMS = (float)CLHEP::RandGauss::shoot(20., 4.);
    float emptyFraction = (float)CLHEP::RandGauss::shoot(.5, .2);

    performanceSummary.setNumberOfDigis(*iDet, nDigisMean, nDigisRMS, emptyFraction);  // set values
  }
  clock_t presentTime = clock();
  performanceSummary.setTimeStamp((unsigned long long)presentTime);
  performanceSummary.print();

  edm::Service<cond::service::PoolDBOutputService> poolDBService;  // write to DB
  if (poolDBService.isAvailable()) {
    try {
      if (poolDBService->isNewTagRequest("SiPixelPerformanceSummaryRcd")) {
        edm::LogInfo("Tag") << " is a new tag request.";
        poolDBService->createOneIOV<SiPixelPerformanceSummary>(
            performanceSummary, poolDBService->beginOfTime(), "SiPixelPerformanceSummaryRcd");
      } else {
        edm::LogInfo("Tag") << " tag exists already";
        poolDBService->appendOneIOV<SiPixelPerformanceSummary>(
            performanceSummary, poolDBService->currentTime(), "SiPixelPerformanceSummaryRcd");
      }
    } catch (const cond::Exception& err) {
      edm::LogError("DBWriting") << err.what() << std::endl;
    } catch (const std::exception& err) {
      edm::LogError("DBWriting") << "caught std::exception " << err.what() << std::endl;
    } catch (...) {
      edm::LogError("DBWriting") << "unknown error" << std::endl;
    }
  } else
    edm::LogError("PoolDBOutputService") << "service unavailable" << std::endl;
}
DEFINE_FWK_MODULE(SiPixelPerformanceSummaryBuilder);
