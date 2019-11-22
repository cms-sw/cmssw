#include <sys/time.h>

#include "CLHEP/Random/RandGauss.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "CondTools/SiPixel/test/SiPixelPerformanceSummaryBuilder.h"

using namespace cms;

SiPixelPerformanceSummaryBuilder::SiPixelPerformanceSummaryBuilder(const edm::ParameterSet& iConfig) {}

SiPixelPerformanceSummaryBuilder::~SiPixelPerformanceSummaryBuilder() {}

void SiPixelPerformanceSummaryBuilder::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  edm::LogInfo("SiPixelPerformanceSummaryBuilder") << pDD->detUnits().size() << " detectors" << std::endl;

  for (const auto& it : pDD->detUnits()) {
    if (dynamic_cast<PixelGeomDetUnit const*>(it) != 0) {
      detectorModules_.push_back(it->geographicalId().rawId());
    }
  }
  edm::LogInfo("Modules") << "detectorModules_.size() = " << detectorModules_.size();

  SiPixelPerformanceSummary* performanceSummary = new SiPixelPerformanceSummary();

  for (std::vector<uint32_t>::const_iterator iDet = detectorModules_.begin();  // fill object
       iDet != detectorModules_.end();
       ++iDet) {
    float nDigisMean = (float)CLHEP::RandGauss::shoot(50., 20.);  // generate random values for each detId
    float nDigisRMS = (float)CLHEP::RandGauss::shoot(20., 4.);
    float emptyFraction = (float)CLHEP::RandGauss::shoot(.5, .2);

    performanceSummary->setNumberOfDigis(*iDet, nDigisMean, nDigisRMS, emptyFraction);  // set values
  }
  clock_t presentTime = clock();
  performanceSummary->setTimeStamp((unsigned long long)presentTime);
  performanceSummary->print();

  edm::Service<cond::service::PoolDBOutputService> poolDBService;  // write to DB
  if (poolDBService.isAvailable()) {
    try {
      if (poolDBService->isNewTagRequest("SiPixelPerformanceSummaryRcd")) {
        edm::LogInfo("Tag") << " is a new tag request.";
        poolDBService->createNewIOV<SiPixelPerformanceSummary>(performanceSummary,
                                                               poolDBService->beginOfTime(),
                                                               poolDBService->endOfTime(),
                                                               "SiPixelPerformanceSummaryRcd");
      } else {
        edm::LogInfo("Tag") << " tag exists already";
        poolDBService->appendSinceTime<SiPixelPerformanceSummary>(
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

void SiPixelPerformanceSummaryBuilder::beginJob() {}

void SiPixelPerformanceSummaryBuilder::endJob() {}
