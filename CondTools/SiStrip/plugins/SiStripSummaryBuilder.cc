// system include files
#include <iostream>
#include <fstream>

// user include files
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

class SiStripSummaryBuilder : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripSummaryBuilder(const edm::ParameterSet& iConfig);

  ~SiStripSummaryBuilder() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::FileInPath fp_;
  bool printdebug_;

  edm::ParameterSet iConfig_;
};

SiStripSummaryBuilder::SiStripSummaryBuilder(const edm::ParameterSet& iConfig)
    : fp_(iConfig.getUntrackedParameter<edm::FileInPath>(
          "file", edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)),
      iConfig_(iConfig) {}

void SiStripSummaryBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();
  edm::LogInfo("SiStripSummaryBuilder") << "... creating dummy SiStripSummary Data for Run " << run << "\n "
                                        << std::endl;

  SiStripSummary obj;
  obj.setRunNr(run);

  //* DISCOVER SET OF HISTOGRAMS & QUANTITIES TO BE UPLOADED*//

  std::vector<std::string> userDBContent;
  typedef std::vector<edm::ParameterSet> VParameters;
  VParameters histoList = iConfig_.getParameter<VParameters>("histoList");
  VParameters::iterator ithistoList = histoList.begin();
  VParameters::iterator ithistoListEnd = histoList.end();

  for (; ithistoList != ithistoListEnd; ++ithistoList) {
    std::string keyName = ithistoList->getUntrackedParameter<std::string>("keyName");
    std::vector<std::string> Quantities =
        ithistoList->getUntrackedParameter<std::vector<std::string> >("quantitiesToExtract");
    for (size_t i = 0; i < Quantities.size(); ++i) {
      if (Quantities[i] == "landau") {
        userDBContent.push_back(keyName + std::string("@") + std::string("landauPeak"));
        userDBContent.push_back(keyName + std::string("@") + std::string("landauPeakErr"));
        userDBContent.push_back(keyName + std::string("@") + std::string("landauSFWHM"));
        userDBContent.push_back(keyName + std::string("@") + std::string("landauChi2NDF"));
      } else if (Quantities[i] == "gauss") {
        userDBContent.push_back(keyName + std::string("@") + std::string("gaussMean"));
        userDBContent.push_back(keyName + std::string("@") + std::string("gaussSigma"));
        userDBContent.push_back(keyName + std::string("@") + std::string("gaussChi2NDF"));
      } else if (Quantities[i] == "stat") {
        userDBContent.push_back(keyName + std::string("@") + std::string("entries"));
        userDBContent.push_back(keyName + std::string("@") + std::string("mean"));
        userDBContent.push_back(keyName + std::string("@") + std::string("rms"));
      } else {
        edm::LogError("SiStripSummaryBuilder")
            << "Quantity " << Quantities[i] << " cannot be handled\nAllowed quantities are"
            << "\n  'stat'   that includes: entries, mean, rms"
            << "\n  'landau' that includes: landauPeak, landauPeakErr, landauSFWHM, landauChi2NDF"
            << "\n  'gauss'  that includes: gaussMean, gaussSigma, gaussChi2NDF" << std::endl;
      }
    }
  }
  obj.setUserDBContent(userDBContent);

  std::stringstream ss1;
  ss1 << "QUANTITIES TO BE INSERTED IN DB :"
      << " \n";
  std::vector<std::string> userDBContentA = obj.getUserDBContent();
  for (size_t i = 0; i < userDBContentA.size(); ++i)
    ss1 << userDBContentA[i] << std::endl;
  edm::LogInfo("SiStripSummaryBuilder") << ss1.str();

  //* Loop over detids and create dummy data for each *//

  std::stringstream ss2;
  for (uint32_t detid = 0; detid < 430; detid++) {
    SiStripSummary::InputVector values;
    for (unsigned int i = 0; i < userDBContent.size(); i++)
      values.push_back((float)CLHEP::RandGauss::shoot(50., 30.));

    ss2 << "\n\tdetid " << detid;
    for (size_t j = 0; j < values.size(); ++j)
      ss2 << "\n\t\t " << userDBContent[j] << " " << values[j];

    obj.put(detid, values, userDBContent);

    // See CondFormats/SiStripObjects/SiStripSummary.h for detid definitions

    if (detid == 4)
      detid = 10;
    if (detid == 14)
      detid = 20;
    if (detid == 26)
      detid = 30;
    if (detid == 32)
      detid = 40;
    if (detid == 42)
      detid = 310;
    if (detid == 313)
      detid = 320;
    if (detid == 323)
      detid = 410;
    if (detid == 419)
      detid = 420;
  }

  edm::LogInfo("SiStripSummaryBuilder") << ss2.str();

  //* Insert summary informations in the DB *//

  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripSummaryRcd")) {
      mydbservice->createOneIOV<SiStripSummary>(obj, mydbservice->beginOfTime(), "SiStripSummaryRcd");
    } else {
      mydbservice->appendOneIOV<SiStripSummary>(obj, mydbservice->currentTime(), "SiStripSummaryRcd");
    }
  } else {
    edm::LogError("SiStripSummaryBuilder") << "Service is unavailable" << std::endl;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripSummaryBuilder);
