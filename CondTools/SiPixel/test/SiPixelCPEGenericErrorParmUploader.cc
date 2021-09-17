#include "CondTools/SiPixel/test/SiPixelCPEGenericErrorParmUploader.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>

SiPixelCPEGenericErrorParmUploader::SiPixelCPEGenericErrorParmUploader(const edm::ParameterSet& iConfig)
    : theFileName(iConfig.getParameter<edm::FileInPath>("fileName")),
      theVersion(iConfig.getParameter<double>("version")) {}

SiPixelCPEGenericErrorParmUploader::~SiPixelCPEGenericErrorParmUploader() {}

void SiPixelCPEGenericErrorParmUploader::beginJob() {}

void SiPixelCPEGenericErrorParmUploader::analyze(const edm::Event& iEvent, const edm::EventSetup& setup) {}

void SiPixelCPEGenericErrorParmUploader::endJob() {
  //--- Make the POOL-ORA thingy to store the vector of error structs (DbEntry)
  SiPixelCPEGenericErrorParm* obj = new SiPixelCPEGenericErrorParm;
  obj->fillCPEGenericErrorParm(theVersion, theFileName.fullPath());
  //	std::cout << *obj << std::endl;

  //--- Create a new IOV
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (poolDbService.isAvailable()) {
    if (poolDbService->isNewTagRequest("SiPixelCPEGenericErrorParmRcd"))
      poolDbService->createNewIOV<SiPixelCPEGenericErrorParm>(
          obj, poolDbService->beginOfTime(), poolDbService->endOfTime(), "SiPixelCPEGenericErrorParmRcd");
    else
      poolDbService->appendSinceTime<SiPixelCPEGenericErrorParm>(
          obj, poolDbService->currentTime(), "SiPixelCPEGenericErrorParmRcd");
  } else {
    std::cout << "Pool Service Unavailable" << std::endl;
    // &&& throw an exception???
  }
}
