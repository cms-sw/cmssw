// -*- C++ -*-
//
// Package:    CondFormats/SiPixelObjects
// Class:      SiPixelFEDChannelContainerFromQualityConverter
//
/**\class SiPixelFEDChannelContainerFromQualityConverter SiPixelFEDChannelContainerFromQualityConverter.cc CondFormats/SiPixelObjects/plugins/SiPixelFEDChannelContainerFromQualityConverter.cc
 Description: class to build the SiPixelFEDChannelContainer payloads
*/
//
// Original Author:  Marco Musich
//         Created:  Wed, 27 Nov 2018 12:04:36 GMT
//
//

// system include files
#include <memory>

// user include files
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFEDChannelContainer.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//

class SiPixelFEDChannelContainerFromQualityConverter : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelFEDChannelContainerFromQualityConverter(const edm::ParameterSet&);
  ~SiPixelFEDChannelContainerFromQualityConverter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  SiPixelFEDChannelContainer::SiPixelFEDChannelCollection createFromSiPixelQuality(
      const SiPixelQuality& theQuality, const SiPixelFedCablingMap& theFedCabling);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const std::string m_record;
  const bool printdebug_;
  const bool isMC_;
  const bool removeEmptyPayloads_;
  SiPixelFEDChannelContainer* myQualities;

  int IOVcount_;
  edm::ESWatcher<SiPixelQualityFromDbRcd> SiPixelQualityWatcher_;
};

//
// constructors and destructor
//
SiPixelFEDChannelContainerFromQualityConverter::SiPixelFEDChannelContainerFromQualityConverter(
    const edm::ParameterSet& iConfig)
    : m_record(iConfig.getParameter<std::string>("record")),
      printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)),
      isMC_(iConfig.getUntrackedParameter<bool>("isMC", true)),
      removeEmptyPayloads_(iConfig.getUntrackedParameter<bool>("removeEmptyPayloads", false)) {
  //now do what ever initialization is needed
  myQualities = new SiPixelFEDChannelContainer();
}

SiPixelFEDChannelContainerFromQualityConverter::~SiPixelFEDChannelContainerFromQualityConverter() {
  delete myQualities;
}

//
// member functions
//

// ------------ method called for each event  ------------
void SiPixelFEDChannelContainerFromQualityConverter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  unsigned int RunNumber_ = iEvent.eventAuxiliary().run();
  unsigned int LuminosityBlockNumber_ = iEvent.eventAuxiliary().luminosityBlock();

  bool hasQualityIOV = SiPixelQualityWatcher_.check(iSetup);

  if (hasQualityIOV) {
    //Retrieve the strip quality from conditions
    edm::ESHandle<SiPixelQuality> siPixelQuality_;
    iSetup.get<SiPixelQualityFromDbRcd>().get(siPixelQuality_);

    edm::ESHandle<SiPixelFedCablingMap> cablingMapHandle;
    iSetup.get<SiPixelFedCablingMapRcd>().get(cablingMapHandle);

    std::string scenario = std::to_string(RunNumber_) + "_" + std::to_string(LuminosityBlockNumber_);

    edm::LogInfo("SiPixelFEDChannelContainerFromQualityConverter")
        << "Found IOV:" << RunNumber_ << "(" << LuminosityBlockNumber_ << ")" << std::endl;

    auto theSiPixelFEDChannelCollection =
        this->createFromSiPixelQuality(*(siPixelQuality_.product()), *(cablingMapHandle.product()));

    if (removeEmptyPayloads_ && theSiPixelFEDChannelCollection.empty())
      return;

    myQualities->setScenario(scenario, theSiPixelFEDChannelCollection);

    IOVcount_++;
  }
}

// ------------ method called once each job just before starting event loop  ------------
SiPixelFEDChannelContainer::SiPixelFEDChannelCollection
SiPixelFEDChannelContainerFromQualityConverter::createFromSiPixelQuality(const SiPixelQuality& theQuality,
                                                                         const SiPixelFedCablingMap& theFedCabling) {
  auto fedid_ = theFedCabling.det2fedMap();

  SiPixelFEDChannelContainer::SiPixelFEDChannelCollection theBadChannelCollection;

  auto theDisabledModules = theQuality.getBadComponentList();
  for (const auto& mod : theDisabledModules) {
    //mod.DetID, mod.errorType,mod.BadRocs

    int coded_badRocs = mod.BadRocs;
    std::vector<PixelFEDChannel> disabledChannelsDetSet;
    std::vector<sipixelobjects::CablingPathToDetUnit> path = theFedCabling.pathToDetUnit(mod.DetID);
    auto cabling_ = theFedCabling.cablingTree();
    unsigned int nrocs_inLink(0);
    if (!path.empty()) {
      const sipixelobjects::PixelFEDCabling* aFed = cabling_->fed(path.at(0).fed);
      const sipixelobjects::PixelFEDLink* link = aFed->link(path.at(0).link);
      nrocs_inLink = link->numberOfROCs();
    }

    std::bitset<16> bad_rocs(coded_badRocs);
    unsigned int n_ch = bad_rocs.size() / nrocs_inLink;

    for (unsigned int i_roc = 0; i_roc < n_ch; ++i_roc) {
      unsigned int first_idx = nrocs_inLink * i_roc;
      unsigned int sec_idx = nrocs_inLink * (i_roc + 1) - 1;
      unsigned int mask = pow(2, nrocs_inLink) - 1;
      unsigned int n_setbits = (coded_badRocs >> (i_roc * nrocs_inLink)) & mask;

      if (n_setbits == 0) {
        continue;
      }

      if (n_setbits != mask) {
        edm::LogWarning("SiPixelFEDChannelContainerFromQualityConverter")
            << "Mismatch! DetId: " << mod.DetID << " " << n_setbits << " " << mask << std::endl;
        continue;
      }

      if (printdebug_) {
        edm::LogVerbatim("SiPixelFEDChannelContainerFromQualityConverter") << "passed" << std::endl;
      }

      unsigned int link_id = 99999;
      unsigned int fed_id = 99999;

      for (auto const& p : path) {
        const sipixelobjects::PixelFEDCabling* aFed = cabling_->fed(p.fed);
        const sipixelobjects::PixelFEDLink* link = aFed->link(p.link);
        const sipixelobjects::PixelROC* roc = link->roc(p.roc);
        unsigned int first_roc = roc->idInDetUnit();

        if (first_roc == first_idx) {
          link_id = p.link;
          fed_id = p.fed;
          break;
        }
      }

      if (printdebug_) {
        edm::LogVerbatim("SiPixelFEDChannelContainerFromQualityConverter")
            << " " << fed_id << " " << link_id << " " << first_idx << "  " << sec_idx << std::endl;
      }

      PixelFEDChannel ch = {fed_id, link_id, first_idx, sec_idx};
      disabledChannelsDetSet.push_back(ch);

      if (printdebug_) {
        edm::LogVerbatim("SiPixelFEDChannelContainerFromQualityConverter")
            << i_roc << " " << coded_badRocs << " " << first_idx << " " << sec_idx << std::endl;
        edm::LogVerbatim("SiPixelFEDChannelContainerFromQualityConverter")
            << "=======================================" << std::endl;
      }
    }

    if (!disabledChannelsDetSet.empty()) {
      theBadChannelCollection[mod.DetID] = disabledChannelsDetSet;
    }
  }
  return theBadChannelCollection;
}

void SiPixelFEDChannelContainerFromQualityConverter::beginJob() { IOVcount_ = 0; }

// ------------ method called once each job just after ending the event loop  ------------
void SiPixelFEDChannelContainerFromQualityConverter::endJob() {
  edm::LogInfo("SiPixelFEDChannelContainerFromQualityConverter") << "Analyzed " << IOVcount_ << " IOVs" << std::endl;
  edm::LogInfo("SiPixelFEDChannelContainerFromQualityConverter")
      << "Size of SiPixelFEDChannelContainer object " << myQualities->size() << std::endl
      << std::endl;

  if (printdebug_) {
    edm::LogInfo("SiPixelFEDChannelContainerFromQualityConverter")
        << "Content of SiPixelFEDChannelContainer " << std::endl;

    // use built-in method in the CondFormat
    myQualities->printAll();
  }

  // Form the data here
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    cond::Time_t valid_time = poolDbService->currentTime();
    // this writes the payload to begin in current run defined in cfg
    if (!isMC_) {
      poolDbService->writeOne(myQualities, valid_time, m_record);
    } else {
      // for MC IOV since=1
      poolDbService->writeOne(myQualities, 1, m_record);
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiPixelFEDChannelContainerFromQualityConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Writes payloads of type SiPixelFEDChannelContainer");
  desc.addUntracked<bool>("printDebug", false);
  desc.addUntracked<bool>("removeEmptyPayloads", false);
  desc.add<std::string>("record", "SiPixelStatusScenariosRcd");
  descriptions.add("SiPixelFEDChannelContainerFromQualityConverter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelFEDChannelContainerFromQualityConverter);
