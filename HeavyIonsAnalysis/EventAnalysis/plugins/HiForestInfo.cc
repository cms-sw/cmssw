// -*- C++ -*-
//
// Package:    HiForestInfo
// Class:      HiForestInfo
//
/**\class HiForestInfo HiForestInfo.cc CmsHi/HiForestInfo/src/HiForestInfo.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Fri May 25 06:50:40 EDT 2012
// $Id: HiForestInfo.cc,v 1.1 2012/05/25 11:14:32 yilmaz Exp $
//
//

// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TTree.h"

//
// class declaration
//

class HiForestInfo : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HiForestInfo(const edm::ParameterSet&);
  ~HiForestInfo() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void beginRun(edm::Run const&, edm::EventSetup const&);
  void endRun(edm::Run const&, edm::EventSetup const&);

  // ----------member data ---------------------------

  edm::Service<TFileService> fs_;

  std::vector<std::string> info_;

  TTree* HiForestVersionTree;
  std::string HiForestVersion_;
  std::string GlobalTagLabel_;
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
HiForestInfo::HiForestInfo(const edm::ParameterSet& iConfig) {
  usesResource("TFileService");

  info_ = iConfig.getParameter<std::vector<std::string> >("info");
  HiForestVersion_ = iConfig.getParameter<std::string>("HiForestVersion");
  GlobalTagLabel_ = iConfig.getParameter<std::string>("GlobalTagLabel");
}

HiForestInfo::~HiForestInfo() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void HiForestInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) { using namespace edm; }

// ------------ method called once each job just before starting event loop  ------------
void HiForestInfo::beginJob() {
  HiForestVersionTree = fs_->make<TTree>("HiForest", "HiForest");
  for (uint32_t i = 0; i < info_.size(); ++i)
    HiForestVersionTree->Branch(Form("info_%i", i), info_[i].data(), "info/C");

  HiForestVersionTree->Branch("HiForestVersion", HiForestVersion_.data(), "HiForestVersion/C");
  HiForestVersionTree->Branch("GlobalTag", GlobalTagLabel_.data(), "GlobalTag/C");

  HiForestVersionTree->Fill();
}

// ------------ method called once each job just after ending the event loop  ------------
void HiForestInfo::endJob() {}

// ------------ method called when starting to processes a run  ------------
void HiForestInfo::beginRun(edm::Run const&, edm::EventSetup const&) {}

// ------------ method called when ending the processing of a run  ------------
void HiForestInfo::endRun(edm::Run const&, edm::EventSetup const&) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HiForestInfo::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HiForestInfo);
