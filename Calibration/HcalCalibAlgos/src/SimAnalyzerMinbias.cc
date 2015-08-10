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
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

//#define debugLog

namespace HcalSimMinbias {
  struct myInfo{
    double theMB0, theMB1, theMB2, theMB3, theMB4;
    myInfo() {
      theMB0 = theMB1 = theMB2 = theMB3 = theMB4 = 0;
    }
  };
  struct Counters {
    Counters() {}
    std::string                        fOutputFileName_;
    mutable std::map<HcalDetId,myInfo> myMap_;
  };
}

// class declaration
class SimAnalyzerMinbias : public edm::stream::EDAnalyzer<edm::GlobalCache<HcalSimMinbias::Counters> > {

public:
  explicit SimAnalyzerMinbias(const edm::ParameterSet&, const HcalSimMinbias::Counters* count);
  ~SimAnalyzerMinbias();

  static std::unique_ptr<HcalSimMinbias::Counters> initializeGlobalCache(edm::ParameterSet const& iConfig) {
    HcalSimMinbias::Counters* count = new HcalSimMinbias::Counters();
    count->fOutputFileName_= iConfig.getUntrackedParameter<std::string>("HistOutFile");
    edm::LogInfo("AnalyzerMB") << "Store o/p in " << count->fOutputFileName_;
    return std::unique_ptr<HcalSimMinbias::Counters>(count);
  }

  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endStream() override;
  static  void globalEndJob(const HcalSimMinbias::Counters* counters);
    
private:
 
  // ----------member data ---------------------------
  double      timeCut;
  std::map<HcalDetId,HcalSimMinbias::myInfo> myMap_;
  edm::EDGetTokenT<edm::HepMCProduct>        tok_evt_;
  edm::EDGetTokenT<edm::PCaloHitContainer>   tok_hcal_;
};

// constructors and destructor

SimAnalyzerMinbias::SimAnalyzerMinbias(const edm::ParameterSet& iConfig, 
				       const HcalSimMinbias::Counters* count) {

  // get name of output file with histogramms
  timeCut         = iConfig.getUntrackedParameter<double>("TimeCut", 500);
    
  // get token names of modules, producing object collections
  tok_evt_ = consumes<edm::HepMCProduct>(edm::InputTag("generator"));
  tok_hcal_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","HcalHits"));

  edm::LogInfo("AnalyzerMB") << "Use Time cut of " << timeCut << " ns";
  myMap_.clear();
}
  
SimAnalyzerMinbias::~SimAnalyzerMinbias() { }

void SimAnalyzerMinbias::endStream() {

  for (std::map<HcalDetId,HcalSimMinbias::myInfo>::const_iterator itr=myMap_.begin(); itr != myMap_.end(); ++itr) {
    HcalDetId id = itr->first;
    std::map<HcalDetId,HcalSimMinbias::myInfo>::iterator itr1 = (globalCache()->myMap_).find(id);
    if (itr1 == globalCache()->myMap_.end()) {
      HcalSimMinbias::myInfo info;
      globalCache()->myMap_[id] = info;
      itr1 = (globalCache()->myMap_).find(id);
    }
    itr1->second.theMB0 += itr->second.theMB0;
    itr1->second.theMB1 += itr->second.theMB1;
    itr1->second.theMB2 += itr->second.theMB2;
    itr1->second.theMB3 += itr->second.theMB3;
    itr1->second.theMB4 += itr->second.theMB4;
  }
}
  
void SimAnalyzerMinbias::globalEndJob(const HcalSimMinbias::Counters* count) {
   
  TFile* hOutputFile = new TFile(count->fOutputFileName_.c_str(), "RECREATE") ;
  TTree* myTree      = new TTree("RecJet","RecJet Tree");
  int            mydet, mysubd, depth, iphi, ieta, cells;
  float          mom0_MB, mom1_MB, mom2_MB, mom3_MB, mom4_MB;
  myTree->Branch("mydet",    &mydet, "mydet/I");
  myTree->Branch("mysubd",   &mysubd, "mysubd/I");
  myTree->Branch("cells",    &cells, "cells");
  myTree->Branch("depth",    &depth, "depth/I");
  myTree->Branch("ieta",     &ieta, "ieta/I");
  myTree->Branch("iphi",     &iphi, "iphi/I");
  myTree->Branch("mom0_MB",  &mom0_MB, "mom0_MB/F");
  myTree->Branch("mom1_MB",  &mom1_MB, "mom1_MB/F");
  myTree->Branch("mom2_MB",  &mom2_MB, "mom2_MB/F");
  myTree->Branch("mom3_MB",  &mom2_MB, "mom3_MB/F");
  myTree->Branch("mom4_MB",  &mom4_MB, "mom4_MB/F");

  cells = 0;
  for (std::map<HcalDetId,HcalSimMinbias::myInfo>::const_iterator itr=count->myMap_.begin(); itr != count->myMap_.end(); ++itr) {
    //  std::cout << " Hello me here" << std::endl;
    mysubd = itr->first.subdet();
    depth  = itr->first.depth();
    iphi   = itr->first.iphi();
    ieta   = itr->first.ieta();
    HcalSimMinbias::myInfo info = itr->second;
    if (info.theMB0 > 0) { 
      mom0_MB = info.theMB0;
      mom1_MB = info.theMB1;
      mom2_MB = info.theMB2;
      mom3_MB = info.theMB3;
      mom4_MB = info.theMB4;
      cells++;
      
      edm::LogInfo("AnalyzerMB") << " Result=  " << mysubd << " " << ieta << " "
				 << iphi << " mom0  " << mom0_MB << " mom1 " 
				 << mom1_MB << " mom2 " << mom2_MB << " mom3 "
				 << mom3_MB << " mom4 " << mom4_MB;
      myTree->Fill();
    }
  }
  edm::LogInfo("AnalyzerMB") << "cells " << cells;    
  hOutputFile->cd();
  myTree->Write();
  hOutputFile->Write();   
  hOutputFile->Close() ;
}

// ------------ method called to produce the data  ------------
  
void SimAnalyzerMinbias::analyze(const edm::Event& iEvent, 
				 const edm::EventSetup&) {

  edm::LogInfo("AnalyzerMB") << " Start SimAnalyzerMinbias::analyze " 
			     << iEvent.id().run() << ":" << iEvent.id().event();
  
  edm::Handle<edm::HepMCProduct> evtMC;
  iEvent.getByToken(tok_evt_, evtMC);  
  if (!evtMC.isValid()) {
    edm::LogWarning("AnalyzerMB") << "no HepMCProduct found";
  } else {
    const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
    edm::LogInfo("AnalyzerMB") << "Event with " << myGenEvent->particles_size()
			       << " particles + " << myGenEvent->vertices_size()
			       << " vertices";
  }

 
  edm::Handle<edm::PCaloHitContainer> hcalHits;
  iEvent.getByToken(tok_hcal_,hcalHits);
  if (!hcalHits.isValid()) {
    edm::LogWarning("AnalyzerMB") << "Error! can't get HcalHits product!";
    return ;
  }
  
  const edm::PCaloHitContainer * HitHcal = hcalHits.product () ;
  std::map<HcalDetId,double>  hitMap;
  for (std::vector<PCaloHit>::const_iterator hcalItr = HitHcal->begin (); 
       hcalItr != HitHcal->end(); ++hcalItr) {
    double time      = hcalItr->time();
    if (time < timeCut) {
      double energyhit = hcalItr->energy();
      HcalDetId hid    = HcalDetId(hcalItr->id());
      std::map<HcalDetId,double>::iterator itr1 = hitMap.find(hid);
      if (itr1 == hitMap.end()) {
	hitMap[hid] = 0;
	itr1 = hitMap.find(hid);
      }
       itr1->second += energyhit;
    }
  }
  edm::LogInfo("AnalyzerMB") << "extract information of " << hitMap.size() 
			     << " towers from " << HitHcal->size() << " hits";

  for (std::map<HcalDetId,double>::const_iterator hcalItr=hitMap.begin(); 
       hcalItr != hitMap.end(); ++hcalItr) {
    HcalDetId hid    = hcalItr->first;
    double energyhit = hcalItr->second;
    std::map<HcalDetId,HcalSimMinbias::myInfo>::iterator itr1 = myMap_.find(hid);
    if (itr1 == myMap_.end()) {
      HcalSimMinbias::myInfo info;
      myMap_[hid] = info;
      itr1 = myMap_.find(hid);
    } 
    itr1->second.theMB0++;
    itr1->second.theMB1 += energyhit;
    itr1->second.theMB2 += (energyhit*energyhit);
    itr1->second.theMB3 += (energyhit*energyhit*energyhit);
    itr1->second.theMB4 += (energyhit*energyhit*energyhit*energyhit);
    edm::LogInfo("AnalyzerMB") << "ID " << hid << " with energy " << energyhit;
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimAnalyzerMinbias);

