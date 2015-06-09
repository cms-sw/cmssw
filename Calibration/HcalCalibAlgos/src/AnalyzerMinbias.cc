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
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/Provenance/interface/Provenance.h"

#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"

namespace HcalMinbias {
  struct myInfo{
    double theMB0, theMB1, theMB2, theMB3, theMB4;
    double theNS0, theNS1, theNS2, theNS3, theNS4;
    double theDif0, theDif1, theDif2, runcheck;
    myInfo() {
      theMB0 = theMB1 = theMB2 = theMB3 = theMB4 = 0;
      theNS0 = theNS1 = theNS2 = theNS3 = theNS4 = 0;
      theDif0 = theDif1 = theDif2 = runcheck = 0;
    }
  };
  struct Counters {
    Counters() {}
    std::string                                       fOutputFileName_;
    mutable std::map<std::pair<int,HcalDetId>,myInfo> myMap_;
  };
}

// constructors and destructor
class AnalyzerMinbias : public edm::stream::EDAnalyzer<edm::GlobalCache<HcalMinbias::Counters> > {
public:
  explicit AnalyzerMinbias(const edm::ParameterSet&, const HcalMinbias::Counters* count);
  ~AnalyzerMinbias();

  static std::unique_ptr<HcalMinbias::Counters> initializeGlobalCache(edm::ParameterSet const& iConfig) {
    HcalMinbias::Counters* count = new HcalMinbias::Counters();
    count->fOutputFileName_= iConfig.getUntrackedParameter<std::string>("HistOutFile");
    return std::unique_ptr<HcalMinbias::Counters>(count);
  }

  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endStream() override;
  static  void globalEndJob(const HcalMinbias::Counters* counters);
  
private:
    
  void analyzeHcal(const HcalRespCorrs* myRecalib, 
		   const HBHERecHitCollection& HithbheNS,
		   const HBHERecHitCollection& HithbheMB,
		   const HFRecHitCollection& HithfNS,
		   const HFRecHitCollection& HithfMB, int algoBit, bool fill);

  // ----------member data ---------------------------
  bool           runNZS_ ;
  double         rnnum;
  
  // Root tree members
  std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo> myMap_;
  edm::EDGetTokenT<HBHERecHitCollection>  tok_hbherecoMB_, tok_hbherecoNoise_;
  edm::EDGetTokenT<HFRecHitCollection>    tok_hfrecoMB_,   tok_hfrecoNoise_;
  edm::EDGetTokenT<HORecHitCollection>    tok_horecoMB_,   tok_horecoNoise_;
  bool theRecalib_, ignoreL1_;
  edm::EDGetTokenT<HBHERecHitCollection>  tok_hbheNormal_;
  edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord> tok_hltL1GtMap_;
};

AnalyzerMinbias::AnalyzerMinbias(const edm::ParameterSet& iConfig, 
				 const HcalMinbias::Counters* counters) {

  // get name of output file with histogramms
  
  
  // get token names of modules, producing object collections
  tok_hbherecoMB_   = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputMB"));
  tok_horecoMB_     = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInputMB"));
  tok_hfrecoMB_     = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputMB"));

  tok_hbherecoNoise_= consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputNoise"));
  tok_horecoNoise_  = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInputNoise"));
  tok_hfrecoNoise_  = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputNoise"));

  theRecalib_       = iConfig.getParameter<bool>("Recalib"); 
  ignoreL1_         = iConfig.getUntrackedParameter<bool>("IgnoreL1", true);
  runNZS_           = iConfig.getUntrackedParameter<bool>("RunNZS", true);


  tok_hbheNormal_   = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  tok_hltL1GtMap_   = consumes<L1GlobalTriggerObjectMapRecord>(edm::InputTag("hltL1GtObjectMap"));
}
  
AnalyzerMinbias::~AnalyzerMinbias() { }
  
void AnalyzerMinbias::endStream() {

  for (std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo>::const_iterator itr=myMap_.begin(); itr != myMap_.end(); ++itr) {
    std::pair<int,HcalDetId> id = itr->first;
    std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo>::iterator itr1 = (globalCache()->myMap_).find(id);
    if (itr1 == globalCache()->myMap_.end()) {
      HcalMinbias::myInfo info;
      globalCache()->myMap_[id] = info;
      itr1 = (globalCache()->myMap_).find(id);
      itr1->second.runcheck = itr->second.runcheck;
    }
    itr1->second.theNS0  += itr->second.theNS0;
    itr1->second.theNS1  += itr->second.theNS1;
    itr1->second.theNS2  += itr->second.theNS2;
    itr1->second.theNS3  += itr->second.theNS3;
    itr1->second.theNS4  += itr->second.theNS4;
    itr1->second.theMB0  += itr->second.theMB0;
    itr1->second.theMB1  += itr->second.theMB1;
    itr1->second.theMB2  += itr->second.theMB2;
    itr1->second.theMB3  += itr->second.theMB3;
    itr1->second.theMB4  += itr->second.theMB4;
    itr1->second.theDif0 += itr->second.theDif0;
    itr1->second.theDif1 += itr->second.theDif1;
    itr1->second.theDif2 += itr->second.theDif2;
  }
}
  
//  EndJob
//
void AnalyzerMinbias::globalEndJob(const HcalMinbias::Counters* count) {
   
  TFile* hOutputFile_ = new TFile(count->fOutputFileName_.c_str(), "RECREATE") ;
  TTree* myTree_      = new TTree("RecJet","RecJet Tree");
  double         rnnumber;
  int            mydet, mysubd, depth, iphi, ieta, cells, trigbit;
  float          phi, eta;
  float          mom0_MB, mom1_MB, mom2_MB, mom3_MB, mom4_MB, occup;
  float          mom0_Noise, mom1_Noise, mom2_Noise, mom3_Noise, mom4_Noise;
  float          mom0_Diff, mom1_Diff, mom2_Diff;
  myTree_->Branch("mydet",       &mydet,      "mydet/I");
  myTree_->Branch("mysubd",      &mysubd,     "mysubd/I");
  myTree_->Branch("cells",       &cells,      "cells");
  myTree_->Branch("depth",       &depth,      "depth/I");
  myTree_->Branch("ieta",        &ieta,       "ieta/I");
  myTree_->Branch("iphi",        &iphi,       "iphi/I");
  myTree_->Branch("eta",         &eta,        "eta/F");
  myTree_->Branch("phi",         &phi,        "phi/F");
  myTree_->Branch("mom0_MB",     &mom0_MB,    "mom0_MB/F");
  myTree_->Branch("mom1_MB",     &mom1_MB,    "mom1_MB/F");
  myTree_->Branch("mom2_MB",     &mom2_MB,    "mom2_MB/F");
  myTree_->Branch("mom3_MB",     &mom3_MB,    "mom3_MB/F");
  myTree_->Branch("mom4_MB",     &mom4_MB,    "mom4_MB/F");
  myTree_->Branch("mom0_Noise",  &mom0_Noise, "mom0_Noise/F");
  myTree_->Branch("mom1_Noise",  &mom1_Noise, "mom1_Noise/F");
  myTree_->Branch("mom2_Noise",  &mom2_Noise, "mom2_Noise/F");
  myTree_->Branch("mom3_Noise",  &mom2_Noise, "mom3_Noise/F");
  myTree_->Branch("mom4_Noise",  &mom4_Noise, "mom4_Noise/F");
  myTree_->Branch("mom0_Diff",   &mom0_Diff,  "mom0_Diff/F");
  myTree_->Branch("mom1_Diff",   &mom1_Diff,  "mom1_Diff/F");
  myTree_->Branch("mom2_Diff",   &mom2_Diff,  "mom2_Diff/F");
  myTree_->Branch("occup",       &occup,      "occup/F");
  myTree_->Branch("trigbit",     &trigbit,    "trigbit/I");
  myTree_->Branch("rnnumber",    &rnnumber,   "rnnumber/D");

  int ii=0;
  for (std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo>::const_iterator itr=count->myMap_.begin(); itr != count->myMap_.end(); ++itr) {
    LogDebug("AnalyzerMB") << "Fired trigger bit number " << itr->first.first;
    HcalMinbias::myInfo info = itr->second;
    if (info.theMB0 > 0) { 
      mom0_MB    = info.theMB0;
      mom1_MB    = info.theMB1;
      mom2_MB    = info.theMB2;
      mom3_MB    = info.theMB3;
      mom4_MB    = info.theMB4;
      mom0_Noise = info.theNS0;
      mom1_Noise = info.theNS1;
      mom2_Noise = info.theNS2;
      mom3_Noise = info.theNS3;
      mom4_Noise = info.theNS4;
      mom0_Diff  = info.theDif0;
      mom1_Diff  = info.theDif1;
      mom2_Diff  = info.theDif2;
      rnnumber   = info.runcheck;
      trigbit    = itr->first.first; 
      mysubd     = itr->first.second.subdet();
      depth      = itr->first.second.depth();
      ieta       = itr->first.second.ieta();
      iphi       = itr->first.second.iphi();
      
      LogDebug("AnalyzerMB") << " Result=  " << trigbit << " " << mysubd
			     << " " << ieta << " " << iphi << " mom0  "
			     << mom0_MB << " mom1 " << mom1_MB << " mom2 "
			     << mom2_MB << " mom3 " << mom3_MB << " mom4 " 
			     << mom4_MB << " mom0_Noise " << mom0_Noise 
			     << " mom1_Noise " << mom1_Noise << " mom2_Noise "
			     << mom2_Noise << " mom3_Noise " << mom3_Noise 
			     << " mom4_Noise " << mom4_Noise << " mom0_Diff "
			     << mom0_Diff << " mom1_Diff " << mom1_Diff
			     << " mom2_Diff " << mom2_Diff;
      myTree_->Fill();
      ii++;
    }
  }
  cells = ii; 
  LogDebug("AnalyzerMB") << "cells" << " " << cells;    
  hOutputFile_->Write();   
  hOutputFile_->cd();
  myTree_->Write();
  hOutputFile_->Close() ;
}

//
// member functions
//
  
// ------------ method called to produce the data  ------------
  
void AnalyzerMinbias::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    
  rnnum = (float)iEvent.run(); 
  const HcalRespCorrs* myRecalib=0;
  if (theRecalib_) {
    edm::ESHandle <HcalRespCorrs> recalibCorrs;
    iSetup.get<HcalRespCorrsRcd>().get("recalibrate",recalibCorrs);
    myRecalib = recalibCorrs.product();
  } // theRecalib
    
  edm::Handle<HBHERecHitCollection> hbheNormal;
  iEvent.getByToken(tok_hbheNormal_, hbheNormal);
  if (!hbheNormal.isValid()) {  
    edm::LogInfo("AnalyzerMB") << " hbheNormal failed";
  } else {
    edm::LogInfo("AnalyzerMB") << " The size of the normal collection "
			       << hbheNormal->size();
  }

  edm::Handle<HBHERecHitCollection> hbheNS;
  iEvent.getByToken(tok_hbherecoNoise_, hbheNS);
  if (!hbheNS.isValid()) {
    edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbheNoise product!";
    return ;
  }
  const HBHERecHitCollection HithbheNS = *(hbheNS.product());
  edm::LogInfo("AnalyzerMB") << "HBHE NS size of collection " << HithbheNS.size();
  if (runNZS_  && HithbheNS.size() != 5184) {
    edm::LogWarning("AnalyzerMB") << "HBHE NS problem " << rnnum << " size "
				  << HithbheNS.size();
    return;
  }
  
  edm::Handle<HBHERecHitCollection> hbheMB;
  iEvent.getByToken(tok_hbherecoMB_, hbheMB);
  if (!hbheMB.isValid()) {
    edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe product!";
    return ;
  }
  const HBHERecHitCollection HithbheMB = *(hbheMB.product());
  edm::LogInfo("AnalyzerMB") << "HBHE MB size of collection " << HithbheMB.size();
  if (runNZS_  && HithbheMB.size() != 5184) {
    edm::LogWarning("AnalyzerMB") << "HBHE problem " << rnnum << " size "
				  << HithbheMB.size();
    return;
  }
    
  edm::Handle<HFRecHitCollection> hfNS;
  iEvent.getByToken(tok_hfrecoNoise_, hfNS);
  if (!hfNS.isValid()) {
    edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hfNoise product!";
    return ;
  }
  const HFRecHitCollection HithfNS = *(hfNS.product());
  edm::LogInfo("AnalyzerMB") << "HF NS size of collection "<< HithfNS.size();
  if (runNZS_  && HithfNS.size() != 1728) {
    edm::LogWarning("AnalyzerMB") << "HF NS problem " << rnnum << " size "
				  << HithfNS.size();
    return;
  }
  
  edm::Handle<HFRecHitCollection> hfMB;
  iEvent.getByToken(tok_hfrecoMB_, hfMB);
  if (!hfMB.isValid()) {
    edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hf product!";
    return ;
  }
  const HFRecHitCollection HithfMB = *(hfMB.product());
  edm::LogInfo("AnalyzerMB") << "HF MB size of collection " << HithfMB.size();
  if(runNZS_  && HithfMB.size() != 1728) {
    edm::LogWarning("AnalyzerMB") << "HF problem " << rnnum << " size "
				  << HithfMB.size();
    return;
  }
    
  if (ignoreL1_) {
    analyzeHcal(myRecalib,HithbheNS,HithbheMB,HithfNS,HithfMB,1,true);
  } else {
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByToken(tok_hltL1GtMap_, gtObjectMapRecord);
    if (gtObjectMapRecord.isValid()) {
      const std::vector<L1GlobalTriggerObjectMap>& objMapVec = gtObjectMapRecord->gtObjectMap();
      int  ii(0);
      bool ok(false), fill(true);
      for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
	   itMap != objMapVec.end(); ++itMap, ++ii) {
	bool resultGt = (*itMap).algoGtlResult();
	if (resultGt == 1) {
	  ok = true;
	  int algoBit = (*itMap).algoBitNumber();
	  analyzeHcal(myRecalib,HithbheNS,HithbheMB,HithfNS,HithfMB,algoBit,fill);
	  fill = false;
	  std::string algoNameStr = (*itMap).algoName();
	  LogDebug("AnalyzerMB") << "Trigger[" << ii << "] " << algoNameStr
				 << " bit " << algoBit << " entered";
	}
      }
      if (!ok) edm::LogInfo("AnalyzerMB") << "No passed L1 Triggers";
    }
  }
}

void AnalyzerMinbias::analyzeHcal(const HcalRespCorrs* myRecalib, 
				  const HBHERecHitCollection& HithbheNS,
				  const HBHERecHitCollection& HithbheMB,
				  const HFRecHitCollection& HithfNS,
				  const HFRecHitCollection& HithfMB,
				  int algoBit, bool fill) {

  // Noise part for HB HE
  std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo> tmpMap;
  tmpMap.clear();
  
  for (HBHERecHitCollection::const_iterator hbheItr=HithbheNS.begin(); 
       hbheItr!=HithbheNS.end(); hbheItr++) {
    
    // Recalibration of energy
    float icalconst=1.;	 
    DetId mydetid = hbheItr->id().rawId();
    if (theRecalib_) icalconst=myRecalib->getValues(mydetid)->getValue();
    
    HBHERecHit aHit(hbheItr->id(),hbheItr->energy()*icalconst,hbheItr->time());
    double energyhit = aHit.energy();
    
    DetId id = (*hbheItr).detid(); 
    HcalDetId hid=HcalDetId(id);
    std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo>::iterator itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
    if (itr1 == myMap_.end()) {
      HcalMinbias::myInfo info;
      myMap_[std::pair<int,HcalDetId>(algoBit,hid)] = info;
      itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
    }
    itr1->second.theNS0++;
    itr1->second.theNS1 += energyhit;
    itr1->second.theNS2 += (energyhit*energyhit);
    itr1->second.theNS3 += (energyhit*energyhit*energyhit);
    itr1->second.theNS4 += (energyhit*energyhit*energyhit*energyhit);
    itr1->second.runcheck = rnnum;
    std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo>::iterator itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
    if (itr2 == tmpMap.end()) {
      HcalMinbias::myInfo info;
      tmpMap[std::pair<int,HcalDetId>(algoBit,hid)] = info;
      itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
    }
    itr2->second.theNS0++;
    itr2->second.theNS1 += energyhit;
    itr2->second.theNS2 += (energyhit*energyhit);
    itr2->second.theNS3 += (energyhit*energyhit*energyhit);
    itr2->second.theNS4 += (energyhit*energyhit*energyhit*energyhit);
    itr2->second.runcheck = rnnum;
      
  } // HBHE_NS
  
    // Signal part for HB HE
  
  for (HBHERecHitCollection::const_iterator hbheItr=HithbheMB.begin(); 
       hbheItr!=HithbheMB.end(); hbheItr++) {
    // Recalibration of energy
    float icalconst=1.;	 
    DetId mydetid = hbheItr->id().rawId();
    if (theRecalib_) icalconst=myRecalib->getValues(mydetid)->getValue();
    
    HBHERecHit aHit(hbheItr->id(),hbheItr->energy()*icalconst,hbheItr->time());
    double energyhit = aHit.energy();
    
    DetId id = (*hbheItr).detid(); 
    HcalDetId hid=HcalDetId(id);
	      
    std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo>::iterator itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
    std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo>::iterator itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	      
    if (itr1 == myMap_.end()) {
      HcalMinbias::myInfo info;
      myMap_[std::pair<int,HcalDetId>(algoBit,hid)] = info;
      itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
    } 
    itr1->second.theMB0++;
    itr1->second.theDif0 = 0;
    itr1->second.theMB1 += energyhit;
    itr1->second.theMB2 += (energyhit*energyhit);
    itr1->second.theMB3 += (energyhit*energyhit*energyhit);
    itr1->second.theMB4 += (energyhit*energyhit*energyhit*energyhit);
    itr1->second.runcheck = rnnum;
    float mydiff = 0.0;
    if (itr2 !=tmpMap.end()) {
      mydiff = energyhit - (itr2->second.theNS1);
      itr1->second.theDif0++;
      itr1->second.theDif1 += mydiff;
      itr1->second.theDif2 += (mydiff*mydiff);
    }
  } // HBHE_MB
	  
    // HF
  
  for (HFRecHitCollection::const_iterator hbheItr=HithfNS.begin(); 
       hbheItr!=HithfNS.end(); hbheItr++) {
    // Recalibration of energy
    float icalconst=1.;	 
    DetId mydetid = hbheItr->id().rawId();
    if (theRecalib_) icalconst=myRecalib->getValues(mydetid)->getValue();
    
    HFRecHit aHit(hbheItr->id(),hbheItr->energy()*icalconst,hbheItr->time());
    double energyhit = aHit.energy();
    // Remove PMT hits
    if(fabs(energyhit) > 40. ) continue;
    DetId id = (*hbheItr).detid(); 
    HcalDetId hid=HcalDetId(id);
    
    std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo>::iterator itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
    
    if (itr1 == myMap_.end()) {
      HcalMinbias::myInfo info;
      myMap_[std::pair<int,HcalDetId>(algoBit,hid)] = info;
      itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
    } 
    itr1->second.theNS0++;
    itr1->second.theNS1 += energyhit;
    itr1->second.theNS2 += (energyhit*energyhit);
    itr1->second.theNS3 += (energyhit*energyhit*energyhit);
    itr1->second.theNS4 += (energyhit*energyhit*energyhit*energyhit);
    itr1->second.runcheck = rnnum;
    std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo>::iterator itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
    if (itr2 == tmpMap.end()) {
      HcalMinbias::myInfo info;
      tmpMap[std::pair<int,HcalDetId>(algoBit,hid)] = info;
      itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
    }
    itr2->second.theNS0++;
    itr2->second.theNS1 += energyhit;
    itr2->second.theNS2 += (energyhit*energyhit);
    itr2->second.theNS3 += (energyhit*energyhit*energyhit);
    itr2->second.theNS4 += (energyhit*energyhit*energyhit*energyhit);
    itr2->second.runcheck = rnnum;
    
  } // HF_NS
	  
	  
    // Signal part for HF
	  
  for (HFRecHitCollection::const_iterator hbheItr=HithfMB.begin(); 
       hbheItr!=HithfMB.end(); hbheItr++) {
    // Recalibration of energy
    float icalconst=1.;	 
    DetId mydetid = hbheItr->id().rawId();
    if (theRecalib_) icalconst=myRecalib->getValues(mydetid)->getValue();
    HFRecHit aHit(hbheItr->id(),hbheItr->energy()*icalconst,hbheItr->time());
    
    double energyhit = aHit.energy();
    // Remove PMT hits
    if(fabs(energyhit) > 40. ) continue;
	      
    DetId id = (*hbheItr).detid(); 
    HcalDetId hid=HcalDetId(id);
	      
    std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo>::iterator itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
    std::map<std::pair<int,HcalDetId>,HcalMinbias::myInfo>::iterator itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
    
    if (itr1 == myMap_.end()) {
      HcalMinbias::myInfo info;
      myMap_[std::pair<int,HcalDetId>(algoBit,hid)] = info;
      itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
    }
    itr1->second.theMB0++;
    itr1->second.theDif0 = 0;
    itr1->second.theMB1 += energyhit;
    itr1->second.theMB2 += (energyhit*energyhit);
    itr1->second.theMB3 += (energyhit*energyhit*energyhit);
    itr1->second.theMB4 += (energyhit*energyhit*energyhit*energyhit);
    itr1->second.runcheck = rnnum;
    float mydiff = 0.0;
    if (itr2 !=tmpMap.end()) {
      mydiff = energyhit - (itr2->second.theNS1);
      itr1->second.theDif0++;
      itr1->second.theDif1 += mydiff;
      itr1->second.theDif2 += (mydiff*mydiff);
    }
  }
}

//define this as a plug-in                                                      
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AnalyzerMinbias);
