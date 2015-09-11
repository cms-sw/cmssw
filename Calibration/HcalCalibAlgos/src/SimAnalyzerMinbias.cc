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
#include "FWCore/Framework/interface/EDAnalyzer.h"
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

// class declaration
class SimAnalyzerMinbias : public edm::EDAnalyzer {

public:
  explicit SimAnalyzerMinbias(const edm::ParameterSet&);
  ~SimAnalyzerMinbias();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob() ;
  virtual void endJob() ;
  virtual void beginRun(const edm::Run& r, const edm::EventSetup& iSetup);
  virtual void endRun(const edm::Run& r, const edm::EventSetup& iSetup);
    
private:
    
  // ----------member data ---------------------------
  std::string fOutputFileName ;
  double      timeCut;
  TFile*      hOutputFile ;
  TTree*      myTree;
  
  // Root tree members
  int   mydet, mysubd, depth, iphi, ieta, cells;
  float mom0_MB, mom1_MB, mom2_MB, mom3_MB, mom4_MB;
  struct myInfo{
    double theMB0, theMB1, theMB2, theMB3, theMB4;
    void MyInfo() {
      theMB0 = theMB1 = theMB2 = theMB3 = theMB4 = 0;
    }
  };
  std::map<HcalDetId,myInfo>               myMap;
  edm::EDGetTokenT<edm::HepMCProduct>      tok_evt_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hcal_;
};

// constructors and destructor

SimAnalyzerMinbias::SimAnalyzerMinbias(const edm::ParameterSet& iConfig) {
  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<std::string>("HistOutFile", "simOutput.root");
  timeCut         = iConfig.getUntrackedParameter<double>("TimeCut", 500);
    
  // get token names of modules, producing object collections
  tok_evt_ = consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"));
  tok_hcal_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","HcalHits"));

#ifdef debugLog
  std::cout << "Use Time cut of " << timeCut << " ns and store o/p in "
	    << fOutputFileName << std::endl;
#endif
}
  
SimAnalyzerMinbias::~SimAnalyzerMinbias() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}
  
void SimAnalyzerMinbias::beginRun( const edm::Run& r, const edm::EventSetup& iSetup) {
}
  
void SimAnalyzerMinbias::endRun( const edm::Run& r, const edm::EventSetup& iSetup) {
}
  
void SimAnalyzerMinbias::beginJob() {
  hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;

  myTree = new TTree("SimJet","SimJet Tree");
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

  myMap.clear();
  return ;
}
  
//  EndJob
//
void SimAnalyzerMinbias::endJob() {
   
  cells = 0;
  // std::cout << " Hello me here" << std::endl;
  for (std::map<HcalDetId,myInfo>::const_iterator itr=myMap.begin(); 
       itr != myMap.end(); ++itr) {
    //  std::cout << " Hello me here" << std::endl;
    mysubd = itr->first.subdet();
    depth  = itr->first.depth();
    iphi   = itr->first.iphi();
    ieta   = itr->first.ieta();
    myInfo info = itr->second;
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
#ifdef debugLog
      std::cout                  << " Result=  " << mysubd << " " << ieta << " "
				 << iphi << " mom0  " << mom0_MB << " mom1 " 
				 << mom1_MB << " mom2 " << mom2_MB << " mom3 "
				 << mom3_MB << " mom4 " << mom4_MB << std::endl;
#endif
      myTree->Fill();
    }
  }
  edm::LogInfo("AnalyzerMB") << "cells " << cells;    
#ifdef debugLog
  std::cout                  << "cells " << cells << std::endl;
#endif
  hOutputFile->cd();
  myTree->Write();
  hOutputFile->Write();   
  hOutputFile->Close() ;
  return ;
}

//
// member functions
//
  
// ------------ method called to produce the data  ------------
  
void SimAnalyzerMinbias::analyze(const edm::Event& iEvent, 
				 const edm::EventSetup&) {
#ifdef debugLog    
  std::cout << " Start SimAnalyzerMinbias::analyze " << iEvent.id().run() 
	    << ":" << iEvent.id().event() << std::endl;
#endif
  
  edm::Handle<edm::HepMCProduct> evtMC;
  iEvent.getByToken(tok_evt_, evtMC);  
  if (!evtMC.isValid()) {
    edm::LogInfo("AnalyzerMB") << "no HepMCProduct found";
#ifdef debugLog
    std::cout                  << "no HepMCProduct found" << std::endl;
#endif
  } else {
    const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
    edm::LogInfo("AnalyzerMB") << "Event with " << myGenEvent->particles_size()
			       << " particles + " << myGenEvent->vertices_size()
			       << " vertices";
#ifdef debugLog
    std::cout                  << "Event with " << myGenEvent->particles_size()
			       << " particles + " << myGenEvent->vertices_size()
			       << " vertices" << std::endl;
#endif
  }

 
  edm::Handle<edm::PCaloHitContainer> hcalHits;
  iEvent.getByToken(tok_hcal_,hcalHits);
  if (!hcalHits.isValid()) {
    edm::LogInfo("AnalyzerMB") << "Error! can't get HcalHits product!";
#ifdef debugLog
    std::cout                  << "Error! can't get HcalHits product!" 
			       << std::endl;
#endif
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
#ifdef debugLog
  std::cout << "extract information of " << hitMap.size() << " towers from "
	    << HitHcal->size() << " hits" << std::endl;
#endif

  for (std::map<HcalDetId,double>::const_iterator hcalItr=hitMap.begin(); 
       hcalItr != hitMap.end(); ++hcalItr) {
    HcalDetId hid    = hcalItr->first;
    double energyhit = hcalItr->second;
    std::map<HcalDetId,myInfo>::iterator itr1 = myMap.find(hid);
    if (itr1 == myMap.end()) {
      myInfo info;
      myMap[hid] = info;
      itr1 = myMap.find(hid);
    } 
    itr1->second.theMB0++;
    itr1->second.theMB1 += energyhit;
    itr1->second.theMB2 += (energyhit*energyhit);
    itr1->second.theMB3 += (energyhit*energyhit*energyhit);
    itr1->second.theMB4 += (energyhit*energyhit*energyhit*energyhit);
#ifdef debugLog
    std::cout << "ID " << hid << " with energy " << energyhit << std::endl;
#endif
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimAnalyzerMinbias);

