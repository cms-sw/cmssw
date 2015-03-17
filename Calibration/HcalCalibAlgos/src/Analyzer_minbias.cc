// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Calibration/HcalCalibAlgos/interface/Analyzer_minbias.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include <fstream>
#include <sstream>
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

// constructors and destructor
namespace cms{
  Analyzer_minbias::Analyzer_minbias(const edm::ParameterSet& iConfig) {
    // get name of output file with histogramms
    
    fOutputFileName = iConfig.getUntrackedParameter<std::string>("HistOutFile");
    
    // get token names of modules, producing object collections
    tok_hbherecoMB_   = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputMB"));
    tok_horecoMB_     = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInputMB"));
    tok_hfrecoMB_     = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputMB"));

    tok_hbherecoNoise_= consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputNoise"));
    tok_horecoNoise_  = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInputNoise"));
    tok_hfrecoNoise_  = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputNoise"));

    theRecalib = iConfig.getParameter<bool>("Recalib"); 

    tok_hbheNormal_   = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
    tok_hltL1GtMap_   = consumes<L1GlobalTriggerObjectMapRecord>(edm::InputTag("hltL1GtObjectMap"));
  }
  
  Analyzer_minbias::~Analyzer_minbias() { }
  
  void Analyzer_minbias::beginRun( const edm::Run& r, const edm::EventSetup& iSetup) { }
  
  void Analyzer_minbias::endRun( const edm::Run& r, const edm::EventSetup& iSetup) { }
  
  void Analyzer_minbias::beginJob() {

    hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;

    myTree = new TTree("RecJet","RecJet Tree");
    myTree->Branch("mydet",       &mydet,      "mydet/I");
    myTree->Branch("mysubd",      &mysubd,     "mysubd/I");
    myTree->Branch("cells",       &cells,      "cells");
    myTree->Branch("depth",       &depth,      "depth/I");
    myTree->Branch("ieta",        &ieta,       "ieta/I");
    myTree->Branch("iphi",        &iphi,       "iphi/I");
    myTree->Branch("eta",         &eta,        "eta/F");
    myTree->Branch("phi",         &phi,        "phi/F");
    myTree->Branch("mom0_MB",     &mom0_MB,    "mom0_MB/F");
    myTree->Branch("mom1_MB",     &mom1_MB,    "mom1_MB/F");
    myTree->Branch("mom2_MB",     &mom2_MB,    "mom2_MB/F");
    myTree->Branch("mom3_MB",     &mom3_MB,    "mom3_MB/F");
    myTree->Branch("mom4_MB",     &mom4_MB,    "mom4_MB/F");
    myTree->Branch("mom0_Noise",  &mom0_Noise, "mom0_Noise/F");
    myTree->Branch("mom1_Noise",  &mom1_Noise, "mom1_Noise/F");
    myTree->Branch("mom2_Noise",  &mom2_Noise, "mom2_Noise/F");
    myTree->Branch("mom3_Noise",  &mom2_Noise, "mom3_Noise/F");
    myTree->Branch("mom4_Noise",  &mom4_Noise, "mom4_Noise/F");
    myTree->Branch("mom0_Diff",   &mom0_Diff,  "mom0_Diff/F");
    myTree->Branch("mom1_Diff",   &mom1_Diff,  "mom1_Diff/F");
    myTree->Branch("mom2_Diff",   &mom2_Diff,  "mom2_Diff/F");
    myTree->Branch("occup",       &occup,      "occup/F");
    myTree->Branch("trigbit",     &trigbit,    "trigbit/I");
    myTree->Branch("rnnumber",    &rnnumber,   "rnnumber/D");

    myMap.clear();
    return ;
  }
  
  //  EndJob
  //
  void Analyzer_minbias::endJob() {
   
    int ii=0;
    for (std::map<std::pair<int,HcalDetId>,myInfo>::const_iterator itr=myMap.begin(); itr != myMap.end(); ++itr) {
      int h = itr->first.first;
      LogDebug("AnalyzerMB") << "Fired trigger bit number " << h; 
      int i = itr->first.second.subdet();
      int j = itr->first.second.depth();
      int k = itr->first.second.iphi();
      int l = itr->first.second.ieta();
      myInfo info = itr->second;
      if (info.theMB0 > 0) { 
	mom0_MB = info.theMB0;
	mom1_MB = info.theMB1;
	mom2_MB = info.theMB2;
	mom3_MB = info.theMB3;
	mom4_MB = info.theMB4;
	mom0_Noise = info.theNS0;
	mom1_Noise = info.theNS1;
	mom2_Noise = info.theNS2;
	mom3_Noise = info.theNS3;
	mom4_Noise = info.theNS4;
	mom0_Diff = info.theDif0;
	mom1_Diff = info.theDif1;
	mom2_Diff = info.theDif2;
	rnnumber = info.runcheck;
	trigbit= h; 
	mysubd = i;
	depth = j;
	ieta = l;
	iphi = k;
	
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
	myTree->Fill();
	ii++;
      }
    }
    cells=ii; 
    LogDebug("AnalyzerMB") << "cells" << " " << cells;    
    hOutputFile->Write();   
    hOutputFile->cd();
    myTree->Write();
    hOutputFile->Close() ;
    return ;
  }
  //
  // member functions
  //
  
  // ------------ method called to produce the data  ------------
  
  void Analyzer_minbias::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    
    rnnum = (float)iEvent.run(); 
    const HcalRespCorrs* myRecalib=0;
    if (theRecalib ) {
      edm::ESHandle <HcalRespCorrs> recalibCorrs;
      iSetup.get<HcalRespCorrsRcd>().get("recalibrate",recalibCorrs);
      myRecalib = recalibCorrs.product();
    } // theRecalib
    
      // Noise part for HB HE
    
    std::map<std::pair<int,HcalDetId>,myInfo> tmpMap;
    tmpMap.clear();
    
    edm::Handle<HBHERecHitCollection> hbheNormal;
    iEvent.getByToken(tok_hbheNormal_, hbheNormal);
    if (!hbheNormal.isValid()) {  
      edm::LogInfo("AnalyzerMB") << " hbheNormal failed";
    } else {
      LogDebug("AnalyzerMB") << " The size of the normal collection "
			     << hbheNormal->size();
    }
    edm::Handle<HBHERecHitCollection> hbheNS;
    iEvent.getByToken(tok_hbherecoNoise_, hbheNS);
    if (!hbheNS.isValid()) {
      edm::LogInfo("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe product!";
      return ;
    }
    
    const HBHERecHitCollection HithbheNS = *(hbheNS.product());
    LogDebug("AnalyzerMB") << "HBHE NS size of collection " << HithbheNS.size();
    if (HithbheNS.size() < 5100) {
      edm::LogInfo("AnalyzerMB") << "HBHE problem " << rnnum << " size "
				 << HithbheNS.size();
      return;
    }
    
    edm::Handle<HBHERecHitCollection> hbheMB;
    iEvent.getByToken(tok_hbherecoMB_, hbheMB);
    
    if (!hbheMB.isValid()) {
      edm::LogInfo("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe product!";
      return ;
    }
    
    const HBHERecHitCollection HithbheMB = *(hbheMB.product());
    LogDebug("AnalyzerMB") << "HBHE MB size of collection " << HithbheMB.size();
    if(HithbheMB.size() < 5100) {
      edm::LogInfo("AnalyzerMB") << "HBHE problem " << rnnum << " size "
				 << HithbheMB.size();
      return;
    }
    
    edm::Handle<HFRecHitCollection> hfNS;
    iEvent.getByToken(tok_hfrecoNoise_, hfNS);
    
    if (!hfNS.isValid()) {
      edm::LogInfo("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe product!";
      return ;
    }
    
    const HFRecHitCollection HithfNS = *(hfNS.product());
    LogDebug("AnalyzerMB") << "HF NS size of collection "<< HithfNS.size();
    if (HithfNS.size() < 1700) {
      edm::LogInfo("AnalyzerMB") << "HF problem " << rnnum << " size "
				 << HithfNS.size();
      return;
    }
    
    edm::Handle<HFRecHitCollection> hfMB;
    iEvent.getByToken(tok_hfrecoMB_, hfMB);
    
    if (!hfMB.isValid()) {
      edm::LogInfo("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe product!";
      return ;
    }
    
    const HFRecHitCollection HithfMB = *(hfMB.product());
    LogDebug("AnalyzerMB") << "HF MB size of collection " << HithfMB.size();
    if(HithfMB.size() < 1700) {
      edm::LogInfo("AnalyzerMB") << "HF problem " << rnnum << " size "
				 <<HithfMB.size();
      return;
    }
    
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByToken(tok_hltL1GtMap_, gtObjectMapRecord);
    
    if (gtObjectMapRecord.isValid()) {
      const std::vector<L1GlobalTriggerObjectMap>& objMapVec = gtObjectMapRecord->gtObjectMap();
      int ii(0);
      bool ok(false);
      for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
	   itMap != objMapVec.end(); ++itMap, ++ii) {
	float algoBit = (*itMap).algoBitNumber();
	bool resultGt = (*itMap).algoGtlResult();
        std::string algoNameStr = (*itMap).algoName();
	
        if (resultGt == 1) {
	  ok = true;
	  for(HBHERecHitCollection::const_iterator hbheItr=HithbheNS.begin(); hbheItr!=HithbheNS.end(); hbheItr++) {
	    
	    // Recalibration of energy
	    float icalconst=1.;	 
	    DetId mydetid = hbheItr->id().rawId();
	    if( theRecalib ) icalconst=myRecalib->getValues(mydetid)->getValue();
	    
	    HBHERecHit aHit(hbheItr->id(),hbheItr->energy()*icalconst,hbheItr->time());
	    
	    double energyhit = aHit.energy();
	    
	    DetId id = (*hbheItr).detid(); 
	    HcalDetId hid=HcalDetId(id);
	    std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	    if (itr1 == myMap.end()) {
	      myInfo info;
	      myMap[std::pair<int,HcalDetId>(algoBit,hid)] = info;
	      itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	    }
	    itr1->second.theNS0++;
	    itr1->second.theNS1 += energyhit;
	    itr1->second.theNS2 += (energyhit*energyhit);
	    itr1->second.theNS4 += (energyhit*energyhit*energyhit*energyhit);
	    itr1->second.runcheck = rnnum;
	    
	    std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	    if (itr2 == tmpMap.end()) {
	      myInfo info;
	      tmpMap[std::pair<int,HcalDetId>(algoBit,hid)] = info;
	      itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	    }
	    itr2->second.theNS0++;
	    itr2->second.theNS1 += energyhit;
	    itr2->second.theNS2 += (energyhit*energyhit);
	    itr2->second.theNS4 += (energyhit*energyhit*energyhit*energyhit);
	    itr2->second.runcheck = rnnum;
	    
	  } // HBHE_NS
	  
	  // Signal part for HB HE
	  
	  for (HBHERecHitCollection::const_iterator hbheItr=HithbheMB.begin(); hbheItr!=HithbheMB.end(); hbheItr++) {
	    // Recalibration of energy
	    float icalconst=1.;	 
	    DetId mydetid = hbheItr->id().rawId();
	    if( theRecalib ) icalconst=myRecalib->getValues(mydetid)->getValue();
	      
	    HBHERecHit aHit(hbheItr->id(),hbheItr->energy()*icalconst,hbheItr->time());
	    double energyhit = aHit.energy();
	      
	    DetId id = (*hbheItr).detid(); 
	    HcalDetId hid=HcalDetId(id);
	      
	    std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	    std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	      
	    if (itr1 == myMap.end()) {
	      myInfo info;
	      myMap[std::pair<int,HcalDetId>(algoBit,hid)] = info;
	      itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
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
	  
	  for (HFRecHitCollection::const_iterator hbheItr=HithfNS.begin(); hbheItr!=HithfNS.end(); hbheItr++) {
	    // Recalibration of energy
	    float icalconst=1.;	 
	    DetId mydetid = hbheItr->id().rawId();
	    if( theRecalib ) icalconst=myRecalib->getValues(mydetid)->getValue();
	      
	    HFRecHit aHit(hbheItr->id(),hbheItr->energy()*icalconst,hbheItr->time());
	    double energyhit = aHit.energy();
	    // Remove PMT hits
	    if(fabs(energyhit) > 40. ) continue;
	      
	    DetId id = (*hbheItr).detid(); 
	    HcalDetId hid=HcalDetId(id);
	      
	    std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	      
	    if (itr1 == myMap.end()) {
	      myInfo info;
	      myMap[std::pair<int,HcalDetId>(algoBit,hid)] = info;
	      itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	    } 
	    itr1->second.theNS0++;
	    itr1->second.theNS1 += energyhit;
	    itr1->second.theNS2 += (energyhit*energyhit);
	    itr1->second.theNS4 += (energyhit*energyhit*energyhit*energyhit);
	    itr1->second.runcheck = rnnum;

	    std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	    if (itr2 == tmpMap.end()) {
	      myInfo info;
	      tmpMap[std::pair<int,HcalDetId>(algoBit,hid)] = info;
	      itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	    }
	    itr2->second.theNS0++;
	    itr2->second.theNS1 += energyhit;
	    itr2->second.theNS2 += (energyhit*energyhit);
	    itr2->second.theNS4 += (energyhit*energyhit*energyhit*energyhit);
	    itr2->second.runcheck = rnnum;
	      
	  } // HF_NS
	  
	  
	  // Signal part for HF
	  
	  for (HFRecHitCollection::const_iterator hbheItr=HithfMB.begin(); hbheItr!=HithfMB.end(); hbheItr++) {
	    // Recalibration of energy
	    float icalconst=1.;	 
	    DetId mydetid = hbheItr->id().rawId();
	    if( theRecalib ) icalconst=myRecalib->getValues(mydetid)->getValue();
	    HFRecHit aHit(hbheItr->id(),hbheItr->energy()*icalconst,hbheItr->time());
	      
	    double energyhit = aHit.energy();
	    // Remove PMT hits
	    if(fabs(energyhit) > 40. ) continue;
	      
	    DetId id = (*hbheItr).detid(); 
	    HcalDetId hid=HcalDetId(id);
	      
	    std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	    std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr2 = tmpMap.find(std::pair<int,HcalDetId>(algoBit,hid));
	      
	    if (itr1 == myMap.end()) {
	      myInfo info;
		myMap[std::pair<int,HcalDetId>(algoBit,hid)] = info;
		itr1 = myMap.find(std::pair<int,HcalDetId>(algoBit,hid));
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
      }
      if (!ok) LogDebug("AnalyzerMB") << "No passed L1 Triggers";
    }
  }
}
