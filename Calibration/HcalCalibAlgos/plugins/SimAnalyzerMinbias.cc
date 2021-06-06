// system include files
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include "TFile.h"
#include "TTree.h"

// class declaration
class SimAnalyzerMinbias : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit SimAnalyzerMinbias(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;
  void beginRun(const edm::Run& r, const edm::EventSetup& iSetup) override{};
  void endRun(const edm::Run& r, const edm::EventSetup& iSetup) override{};

private:
  // ----------member data ---------------------------
  edm::Service<TFileService> fs_;
  double timeCut_;
  TTree* myTree_;

  // Root tree members
  int mydet, mysubd, depth, iphi, ieta, cells;
  float mom0_MB, mom1_MB, mom2_MB, mom3_MB, mom4_MB;
  struct myInfo {
    double theMB0, theMB1, theMB2, theMB3, theMB4;
    myInfo() { theMB0 = theMB1 = theMB2 = theMB3 = theMB4 = 0; }
  };
  std::map<HcalDetId, myInfo> myMap_;
  edm::EDGetTokenT<edm::HepMCProduct> tok_evt_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hcal_;
};

// constructors and destructor

SimAnalyzerMinbias::SimAnalyzerMinbias(const edm::ParameterSet& iConfig) {
  usesResource(TFileService::kSharedResource);
  timeCut_ = iConfig.getUntrackedParameter<double>("TimeCut", 500);

  // get token names of modules, producing object collections
  tok_evt_ = consumes<edm::HepMCProduct>(edm::InputTag("generator"));
  tok_hcal_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "HcalHits"));

  edm::LogVerbatim("AnalyzerMB") << "Use Time cut of " << timeCut_ << " ns";
}

void SimAnalyzerMinbias::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<double>("TimeCut", 500);
  descriptions.add("simAnalyzerMinbias", desc);
}

void SimAnalyzerMinbias::beginJob() {
  myTree_ = fs_->make<TTree>("SimJet", "SimJet Tree");
  myTree_->Branch("mydet", &mydet, "mydet/I");
  myTree_->Branch("mysubd", &mysubd, "mysubd/I");
  myTree_->Branch("cells", &cells, "cells");
  myTree_->Branch("depth", &depth, "depth/I");
  myTree_->Branch("ieta", &ieta, "ieta/I");
  myTree_->Branch("iphi", &iphi, "iphi/I");
  myTree_->Branch("mom0_MB", &mom0_MB, "mom0_MB/F");
  myTree_->Branch("mom1_MB", &mom1_MB, "mom1_MB/F");
  myTree_->Branch("mom2_MB", &mom2_MB, "mom2_MB/F");
  myTree_->Branch("mom3_MB", &mom3_MB, "mom3_MB/F");
  myTree_->Branch("mom4_MB", &mom4_MB, "mom4_MB/F");

  myMap_.clear();
}

//  EndJob
//
void SimAnalyzerMinbias::endJob() {
  cells = 0;
  for (std::map<HcalDetId, myInfo>::const_iterator itr = myMap_.begin(); itr != myMap_.end(); ++itr) {
    mysubd = itr->first.subdet();
    depth = itr->first.depth();
    iphi = itr->first.iphi();
    ieta = itr->first.ieta();
    myInfo info = itr->second;
    if (info.theMB0 > 0) {
      mom0_MB = info.theMB0;
      mom1_MB = info.theMB1;
      mom2_MB = info.theMB2;
      mom3_MB = info.theMB3;
      mom4_MB = info.theMB4;
      cells++;

      edm::LogVerbatim("AnalyzerMB") << " Result=  " << mysubd << " " << ieta << " " << iphi << " mom0  " << mom0_MB
                                     << " mom1 " << mom1_MB << " mom2 " << mom2_MB << " mom3 " << mom3_MB << " mom4 "
                                     << mom4_MB;
      myTree_->Fill();
    }
  }
  edm::LogVerbatim("AnalyzerMB") << "cells " << cells;
}

//
// member functions
//

// ------------ method called to produce the data  ------------

void SimAnalyzerMinbias::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  edm::LogVerbatim("AnalyzerMB") << " Start SimAnalyzerMinbias::analyze " << iEvent.id().run() << ":"
                                 << iEvent.id().event();

  edm::Handle<edm::HepMCProduct> evtMC;
  iEvent.getByToken(tok_evt_, evtMC);
  if (!evtMC.isValid()) {
    edm::LogWarning("AnalyzerMB") << "no HepMCProduct found";
  } else {
    const HepMC::GenEvent* myGenEvent = evtMC->GetEvent();
    edm::LogVerbatim("AnalyzerMB") << "Event with " << myGenEvent->particles_size() << " particles + "
                                   << myGenEvent->vertices_size() << " vertices";
  }

  edm::Handle<edm::PCaloHitContainer> hcalHits;
  iEvent.getByToken(tok_hcal_, hcalHits);
  if (!hcalHits.isValid()) {
    edm::LogWarning("AnalyzerMB") << "Error! can't get HcalHits product!";
    return;
  }

  const edm::PCaloHitContainer* HitHcal = hcalHits.product();
  std::map<HcalDetId, double> hitMap;
  for (std::vector<PCaloHit>::const_iterator hcalItr = HitHcal->begin(); hcalItr != HitHcal->end(); ++hcalItr) {
    double time = hcalItr->time();
    if (time < timeCut_) {
      double energyhit = hcalItr->energy();
      HcalDetId hid = HcalDetId(hcalItr->id());
      std::map<HcalDetId, double>::iterator itr1 = hitMap.find(hid);
      if (itr1 == hitMap.end()) {
        hitMap[hid] = 0;
        itr1 = hitMap.find(hid);
      }
      itr1->second += energyhit;
    }
  }
  edm::LogVerbatim("AnalyzerMB") << "extract information of " << hitMap.size() << " towers from " << HitHcal->size()
                                 << " hits";

  for (std::map<HcalDetId, double>::const_iterator hcalItr = hitMap.begin(); hcalItr != hitMap.end(); ++hcalItr) {
    HcalDetId hid = hcalItr->first;
    double energyhit = hcalItr->second;
    std::map<HcalDetId, myInfo>::iterator itr1 = myMap_.find(hid);
    if (itr1 == myMap_.end()) {
      myInfo info;
      myMap_[hid] = info;
      itr1 = myMap_.find(hid);
    }
    itr1->second.theMB0++;
    itr1->second.theMB1 += energyhit;
    itr1->second.theMB2 += (energyhit * energyhit);
    itr1->second.theMB3 += (energyhit * energyhit * energyhit);
    itr1->second.theMB4 += (energyhit * energyhit * energyhit * energyhit);
    edm::LogVerbatim("AnalyzerMB") << "ID " << hid << " with energy " << energyhit;
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimAnalyzerMinbias);
