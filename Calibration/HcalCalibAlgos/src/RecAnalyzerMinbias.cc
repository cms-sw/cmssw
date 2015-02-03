// system include files
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

// user include files
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

#include "TFile.h"
#include "TTree.h"

#define debugLog

// class declaration
class RecAnalyzerMinbias : public edm::EDAnalyzer {

public:
  explicit RecAnalyzerMinbias(const edm::ParameterSet&);
  ~RecAnalyzerMinbias();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob() ;
  virtual void endJob() ;
    
private:
  void analyzeHcal(const HBHERecHitCollection&, const HFRecHitCollection&,
		   const HcalRespCorrs*, int);
    
  // ----------member data ---------------------------
  std::string fOutputFileName ;
  bool        theRecalib, ignoreL1;
  TFile*      hOutputFile ;
  TTree*      myTree;

  // Root tree members
  double rnnum, rnnumber;
  int    mysubd, depth, iphi, ieta, cells, trigbit;
  float  mom0_MB, mom1_MB, mom2_MB, mom3_MB, mom4_MB;
  struct myInfo{
    double theMB0, theMB1, theMB2, theMB4, runcheck;
    void MyInfo() {
      theMB0 = theMB1 = theMB2 = theMB4 = runcheck = 0;
    }
  };
  std::map<std::pair<int,HcalDetId>,myInfo> myMap;
  edm::EDGetTokenT<HBHERecHitCollection>    tok_hbherecoMB_;
  edm::EDGetTokenT<HFRecHitCollection>      tok_hfrecoMB_;
  edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord> tok_hltL1GtMap_;
};

// constructors and destructor
RecAnalyzerMinbias::RecAnalyzerMinbias(const edm::ParameterSet& iConfig) {

  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<std::string>("HistOutFile");
  theRecalib      = iConfig.getParameter<bool>("Recalib");
  ignoreL1        = iConfig.getUntrackedParameter<bool>("IgnoreL1", false);
    
  // get token names of modules, producing object collections
  tok_hbherecoMB_   = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputMB"));
  tok_hfrecoMB_     = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputMB"));

  tok_hltL1GtMap_   = consumes<L1GlobalTriggerObjectMapRecord>(edm::InputTag("hltL1GtObjectMap"));
}
  
RecAnalyzerMinbias::~RecAnalyzerMinbias() {}
  
void RecAnalyzerMinbias::beginJob() {

  hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;
  myTree = new TTree("RecJet","RecJet Tree");
  myTree->Branch("cells",    &cells,    "cells/I");
  myTree->Branch("mysubd",   &mysubd,   "mysubd/I");
  myTree->Branch("depth",    &depth,    "depth/I");
  myTree->Branch("ieta",     &ieta,     "ieta/I");
  myTree->Branch("iphi",     &iphi,     "iphi/I");
  myTree->Branch("mom0_MB",  &mom0_MB,  "mom0_MB/F");
  myTree->Branch("mom1_MB",  &mom1_MB,  "mom1_MB/F");
  myTree->Branch("mom2_MB",  &mom2_MB,  "mom2_MB/F");
  myTree->Branch("mom4_MB",  &mom4_MB,  "mom4_MB/F");
  myTree->Branch("trigbit",  &trigbit,  "trigbit/I");
  myTree->Branch("rnnumber", &rnnumber, "rnnumber/D");
  myMap.clear();
  return ;
}
  
//  EndJob
//

void RecAnalyzerMinbias::endJob() {
  cells = 0;
  for (std::map<std::pair<int,HcalDetId>,myInfo>::const_iterator itr=myMap.begin(); itr != myMap.end(); ++itr) {
#ifdef debugLog  
    std::cout << "Fired trigger bit number " << itr->first.first << std::endl; 
#endif
    myInfo info = itr->second;
    if (info.theMB0 > 0) { 
      mom0_MB  = info.theMB0;
      mom1_MB  = info.theMB1;
      mom2_MB  = info.theMB2;
      mom4_MB  = info.theMB4;
      rnnumber = info.runcheck;
      trigbit  = itr->first.first;
      mysubd   = itr->first.second.subdet();
      depth    = itr->first.second.depth();
      iphi     = itr->first.second.iphi();
      ieta     = itr->first.second.ieta();
#ifdef debugLog  
      std::cout << " Result=  " << trigbit << " " << mysubd << " " << ieta 
		<< " " << iphi << " mom0  " << mom0_MB << " mom1 " << mom1_MB 
		<< " mom2 " << mom2_MB << " mom4 " << mom4_MB << std::endl;
#endif
      myTree->Fill();
      cells++;
    }
  }
#ifdef debugLog  
  std::cout << "cells" << " " << cells << std::endl;
#endif
  hOutputFile->Write();   
  hOutputFile->cd();
  myTree->Write();
  hOutputFile->Close() ;
}

//
// member functions
//
// ------------ method called to produce the data  ------------
  
void RecAnalyzerMinbias::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    
  rnnum = (float)iEvent.run(); 
  const HcalRespCorrs* myRecalib=0;
  if (theRecalib ) {
    edm::ESHandle <HcalRespCorrs> recalibCorrs;
    iSetup.get<HcalRespCorrsRcd>().get("recalibrate",recalibCorrs);
    myRecalib = recalibCorrs.product();
  } // theRecalib
    
  edm::Handle<HBHERecHitCollection> hbheMB;
  iEvent.getByToken(tok_hbherecoMB_, hbheMB);
  if (!hbheMB.isValid()) {
    edm::LogInfo("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe product!";
    return ;
  }
  const HBHERecHitCollection HithbheMB = *(hbheMB.product());
#ifdef debugLog  
  std::cout << "HBHE MB size of collection " << HithbheMB.size() << std::endl;
#endif
  if (HithbheMB.size() < 5100) {
    edm::LogWarning("AnalyzerMB") << "HBHE problem " << rnnum << " size "
				  << HithbheMB.size();
    return;
  }
    
  edm::Handle<HFRecHitCollection> hfMB;
  iEvent.getByToken(tok_hfrecoMB_, hfMB);
  if (!hfMB.isValid()) {
    edm::LogInfo("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe product!";
    return ;
  }
  const HFRecHitCollection HithfMB = *(hfMB.product());
#ifdef debugLog  
  std::cout << "HF MB size of collection " << HithfMB.size() << std::endl;
#endif
  if (HithfMB.size() < 1700) {
    edm::LogWarning("AnalyzerMB") << "HF problem " << rnnum << " size "
				  << HithfMB.size();
    return;
  }

  if (ignoreL1) {
    analyzeHcal(HithbheMB, HithfMB, myRecalib, 1);
  } else {
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByToken(tok_hltL1GtMap_, gtObjectMapRecord);
    if (gtObjectMapRecord.isValid()) {
      const std::vector<L1GlobalTriggerObjectMap>& objMapVec = gtObjectMapRecord->gtObjectMap();
      bool ok(false);
      for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
	   itMap != objMapVec.end(); ++itMap) {
	bool resultGt = (*itMap).algoGtlResult();
	if (resultGt) {
	  int algoBit = (*itMap).algoBitNumber();
	  analyzeHcal(HithbheMB, HithfMB, myRecalib, algoBit);
	  ok          = true;
	} 
      }
      if (!ok) {
#ifdef debugLog  
	std::cout << "No passed L1 Trigger found" << std::endl;
#endif
      }
    }
  }
}

void RecAnalyzerMinbias::analyzeHcal(const HBHERecHitCollection & HithbheMB,
				     const HFRecHitCollection & HithfMB,
				     const HcalRespCorrs* myRecalib,
				     int algoBit) {
  // Signal part for HB HE
  for (HBHERecHitCollection::const_iterator hbheItr=HithbheMB.begin(); 
       hbheItr!=HithbheMB.end(); hbheItr++) {
    // Recalibration of energy
    float icalconst=1.;	 
    DetId mydetid = hbheItr->id().rawId();
    if (theRecalib) icalconst = myRecalib->getValues(mydetid)->getValue();
    HBHERecHit aHit(hbheItr->id(),hbheItr->energy()*icalconst,hbheItr->time());
    double energyhit = aHit.energy();
    DetId id         = (*hbheItr).detid(); 
    HcalDetId hid    = HcalDetId(id);
	      
    std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
    if (itr1 == myMap.end()) {
      myInfo info;
      myMap[std::pair<int,HcalDetId>(algoBit,hid)] = info;
      itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
    } 
    itr1->second.theMB0++;
    itr1->second.theMB1 += energyhit;
    itr1->second.theMB2 += (energyhit*energyhit);
    itr1->second.theMB4 += (energyhit*energyhit*energyhit*energyhit);
    itr1->second.runcheck = rnnum;
  } // HBHE_MB
 
  // Signal part for HF
  for (HFRecHitCollection::const_iterator hfItr=HithfMB.begin(); 
       hfItr!=HithfMB.end(); hfItr++) {
    // Recalibration of energy
    float icalconst=1.;	 
    DetId mydetid = hfItr->id().rawId();
    if (theRecalib) icalconst = myRecalib->getValues(mydetid)->getValue();
    HFRecHit aHit(hfItr->id(),hfItr->energy()*icalconst,hfItr->time());
    
    double energyhit = aHit.energy();
    DetId id         = (*hfItr).detid(); 
    HcalDetId hid    = HcalDetId(id);
    //
    // Remove PMT hits
    //	 
    if (fabs(energyhit) > 40.) continue;
	      
    std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
    if (itr1 == myMap.end()) {
      myInfo info;
      myMap[std::pair<int,HcalDetId>(algoBit,hid)] = info;
      itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
    }
    itr1->second.theMB0++;
    itr1->second.theMB1 += energyhit;
    itr1->second.theMB2 += (energyhit*energyhit);
    itr1->second.theMB4 += (energyhit*energyhit*energyhit*energyhit);
    itr1->second.runcheck = rnnum;
  }
}

//define this as a plug-in                                                      
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecAnalyzerMinbias);
