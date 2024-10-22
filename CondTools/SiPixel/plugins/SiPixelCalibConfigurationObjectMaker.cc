// -*- C++ -*-
//
// Package:    SiPixelCalibConfigurationObjectMaker
// Class:      SiPixelCalibConfigurationObjectMaker
//
/**\class SiPixelCalibConfigurationObjectMaker SiPixelCalibConfigurationObjectMaker.cc CalibTracker/SiPixelTools/src/SiPixelCalibConfigurationObjectMaker.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Wed Sep 19 13:43:52 CEST 2007
// $Id: SiPixelCalibConfigurationObjectMaker.cc,v 1.5 2009/10/21 15:53:31 heyburn Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <string>
//
// class decleration
//

class SiPixelCalibConfigurationObjectMaker : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelCalibConfigurationObjectMaker(const edm::ParameterSet&);
  ~SiPixelCalibConfigurationObjectMaker() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  std::string inputfilename;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelCalibConfigurationObjectMaker::SiPixelCalibConfigurationObjectMaker(const edm::ParameterSet& iConfig)
    : inputfilename(iConfig.getUntrackedParameter<std::string>(
          "inputFileName", "/afs/cern.ch/cms/Tracker/Pixel/forward/ryd/calib_070106d.dat"))

{
  //now do what ever initialization is needed
  ::putenv((char*)"CORAL_AUTH_USER=testuser");
  ::putenv((char*)"CORAL_AUTH_PASSWORD=test");
}

SiPixelCalibConfigurationObjectMaker::~SiPixelCalibConfigurationObjectMaker() = default;

//
// member functions
//

void SiPixelCalibConfigurationObjectMaker::analyze(const edm::Event&, const edm::EventSetup&) {
  pos::PixelCalibConfiguration fancyCalib(inputfilename);
  SiPixelCalibConfiguration myCalib(fancyCalib);

  std::string fixedmode = fancyCalib.mode();
  std::string tobereplaced = "WithSLink";
  edm::LogPrint("SiPixelCalibConfigurationObjectMaker") << "mode = " << fixedmode << std::endl;
  if (fixedmode.find(tobereplaced) != std::string::npos)
    fixedmode.erase(fixedmode.find(tobereplaced), tobereplaced.length());
  edm::LogPrint("SiPixelCalibConfigurationObjectMaker") << "mode = " << fixedmode << std::endl;
  myCalib.setCalibrationMode(fixedmode);

  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (poolDbService.isAvailable()) {
    if (poolDbService->isNewTagRequest("SiPixelCalibConfigurationRcd")) {
      poolDbService->createOneIOV<SiPixelCalibConfiguration>(
          myCalib, poolDbService->beginOfTime(), "SiPixelCalibConfigurationRcd");
    } else {
      poolDbService->appendOneIOV<SiPixelCalibConfiguration>(
          myCalib, poolDbService->currentTime(), "SiPixelCalibConfigurationRcd");
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelCalibConfigurationObjectMaker);
