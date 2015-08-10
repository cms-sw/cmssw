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
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
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

#include "TH1F.h"
#include "TFile.h"
#include "TTree.h"

namespace HcalRecMinbias {
  struct myInfo{
    double theMB0, theMB1, theMB2, theMB3, theMB4, runcheck;
    myInfo() {
      theMB0 = theMB1 = theMB2 = theMB3 = theMB4 = runcheck = 0;
    }
  };
  struct Counters {
    Counters() {}
    std::string                                       fOutputFileName_;
    mutable std::map<std::pair<int,HcalDetId>,myInfo> myMap_;
  };
}

// class declaration
class RecAnalyzerMinbias : public edm::stream::EDAnalyzer<edm::GlobalCache<HcalRecMinbias::Counters> > {

public:
  explicit RecAnalyzerMinbias(const edm::ParameterSet&, const HcalRecMinbias::Counters* count);
  ~RecAnalyzerMinbias();

  static std::unique_ptr<HcalRecMinbias::Counters> initializeGlobalCache(edm::ParameterSet const& iConfig) {
    HcalRecMinbias::Counters* count = new HcalRecMinbias::Counters();
    count->fOutputFileName_= iConfig.getUntrackedParameter<std::string>("HistOutFile");
    edm::LogInfo("AnalyzerMB") << "Output File: " << count->fOutputFileName_;
    return std::unique_ptr<HcalRecMinbias::Counters>(count);
  }

  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endStream() override;
  static  void globalEndJob(const HcalRecMinbias::Counters* counters);
  virtual void beginJob() ;
    
private:
  void analyzeHcal(const HBHERecHitCollection&, const HFRecHitCollection&, int);
    
  // ----------member data ---------------------------
  bool                       theRecalib_, ignoreL1_, runNZS_;
  double                     eLowHB_, eHighHB_, eLowHE_, eHighHE_;
  double                     eLowHF_, eHighHF_;
  std::map<DetId,double>     corrFactor_;
  std::vector<unsigned int>  hcalID_;
//std::vector<TH1D *>        histo_;
  double                     rnnum;
  std::map<std::pair<int,HcalDetId>,HcalRecMinbias::myInfo> myMap_;
  edm::EDGetTokenT<HBHERecHitCollection>                    tok_hbherecoMB_;
  edm::EDGetTokenT<HFRecHitCollection>                      tok_hfrecoMB_;
  edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord>          tok_hltL1GtMap_;
};

// constructors and destructor
RecAnalyzerMinbias::RecAnalyzerMinbias(const edm::ParameterSet& iConfig, const HcalRecMinbias::Counters* counters) {

  // get name of output file with histogramms
  runNZS_               = iConfig.getParameter<bool>("RunNZS");
  eLowHB_               = iConfig.getParameter<double>("ELowHB");
  eHighHB_              = iConfig.getParameter<double>("EHighHB");
  eLowHE_               = iConfig.getParameter<double>("ELowHE");
  eHighHE_              = iConfig.getParameter<double>("EHighHE");
  eLowHF_               = iConfig.getParameter<double>("ELowHF");
  eHighHF_              = iConfig.getParameter<double>("EHighHF");
  ignoreL1_             = iConfig.getUntrackedParameter<bool>("IgnoreL1",false);
  std::string      cfile= iConfig.getUntrackedParameter<std::string>("CorrFile");
  std::vector<int> ieta = iConfig.getUntrackedParameter<std::vector<int>>("HcalIeta");
  std::vector<int> iphi = iConfig.getUntrackedParameter<std::vector<int>>("HcalIphi");
  std::vector<int> depth= iConfig.getUntrackedParameter<std::vector<int>>("HcalDepth");

  // get token names of modules, producing object collections
  tok_hbherecoMB_   = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputMB"));
  tok_hfrecoMB_     = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputMB"));
  tok_hltL1GtMap_   = consumes<L1GlobalTriggerObjectMapRecord>(edm::InputTag("hltL1GtObjectMap"));

  // Read correction factors
  std::ifstream infile(cfile);
  if (!infile.is_open()) {
    theRecalib_ = false;
    edm::LogInfo("AnalyzerMB") << "Cannot open '" << cfile 
			       << "' for the correction file";
  } else {
    unsigned int ndets(0), nrec(0);
    while(1) {
      unsigned int id;
      double       cfac;
      infile >> id >> cfac;
      if (!infile.good()) break;
      HcalDetId detId(id);
      nrec++;
      std::map<DetId,double>::iterator itr = corrFactor_.find(detId);
      if (itr == corrFactor_.end()) {
	corrFactor_[detId] = cfac;
	ndets++;
      }
    }
    infile.close();
    edm::LogInfo("AnalyzerMB") << "Reads " << nrec << " correction factors for "
			       << ndets << " detIds";
    theRecalib_ = (ndets>0);
  }

  edm::LogInfo("AnalyzerMB") << " Flags (ReCalib): " << theRecalib_
			     << " (IgnoreL1): " << ignoreL1_ << " (NZS) " 
			     << runNZS_ << " and with " << ieta.size() 
			     << " detId for full histogram";
  edm::LogInfo("AnalyzerMB") << "Thresholds for HB " << eLowHB_ << ":" 
			     << eHighHB_ << "  for HE " << eLowHE_ << ":" 
			     << eHighHE_ << "  for HF " << eLowHF_ << ":" 
			     << eHighHF_;
  for (unsigned int k=0; k<ieta.size(); ++k) {
    HcalSubdetector subd = ((std::abs(ieta[k]) > 29) ? HcalForward : 
			    (std::abs(ieta[k]) > 16) ? HcalEndcap :
			    ((std::abs(ieta[k]) == 16) && (depth[k] == 3)) ? HcalEndcap :
			    (depth[k] == 4) ? HcalOuter : HcalBarrel);
    unsigned int id = (HcalDetId(subd,ieta[k],iphi[k],depth[k])).rawId();
    hcalID_.push_back(id);
    edm::LogInfo("AnalyzerMB") << "DetId[" << k << "] " << HcalDetId(id);
  }
  myMap_.clear();
}
  
RecAnalyzerMinbias::~RecAnalyzerMinbias() {}
  
void RecAnalyzerMinbias::beginJob() {
  /*
  std::string hc[5] = {"Empty", "HB", "HE", "HO", "HF"};
  char        name[700], title[700];
  for (unsigned int i=0; i<hcalID_.size(); i++) {
    HcalDetId id = HcalDetId(hcalID_[i]);
    int subdet   = id.subdetId();
    sprintf (name, "%s%d_%d_%d", hc[subdet].c_str(), id.ieta(), id.iphi(), id.depth());
    sprintf (title, "Energy Distribution for %s ieta %d iphi %d depth %d", hc[subdet].c_str(), id.ieta(), id.iphi(), id.depth());
    double xmin = (subdet == 4) ? -10 : -1;
    double xmax = (subdet == 4) ? 90 : 9;
    TH1D*  hh   = new TH1D(name, title, 50, xmin, xmax);
    histo_.push_back(hh);
  };
  */
}

void RecAnalyzerMinbias::endStream() {

  for (std::map<std::pair<int,HcalDetId>,HcalRecMinbias::myInfo>::const_iterator itr=myMap_.begin(); itr != myMap_.end(); ++itr) {
    std::pair<int,HcalDetId> id = itr->first;
    std::map<std::pair<int,HcalDetId>,HcalRecMinbias::myInfo>::iterator itr1 = (globalCache()->myMap_).find(id);
    if (itr1 == globalCache()->myMap_.end()) {
      HcalRecMinbias::myInfo info;
      globalCache()->myMap_[id] = info;
      itr1 = (globalCache()->myMap_).find(id);
      itr1->second.runcheck = itr->second.runcheck;
    }
    itr1->second.theMB0  += itr->second.theMB0;
    itr1->second.theMB1  += itr->second.theMB1;
    itr1->second.theMB2  += itr->second.theMB2;
    itr1->second.theMB3  += itr->second.theMB3;
    itr1->second.theMB4  += itr->second.theMB4;
  }
}
  
//  EndJob
//
void RecAnalyzerMinbias::globalEndJob(const HcalRecMinbias::Counters* count) {
 
  TFile* hOutputFile   = new TFile(count->fOutputFileName_.c_str(), "RECREATE");
  TTree* myTree        = new TTree("RecJet","RecJet Tree");
  double                     rnnumber;
  int                        mysubd, depth, iphi, ieta, cells, trigbit;
  float                      mom0_MB, mom1_MB, mom2_MB, mom3_MB, mom4_MB;
  myTree->Branch("cells",    &cells,    "cells/I");
  myTree->Branch("mysubd",   &mysubd,   "mysubd/I");
  myTree->Branch("depth",    &depth,    "depth/I");
  myTree->Branch("ieta",     &ieta,     "ieta/I");
  myTree->Branch("iphi",     &iphi,     "iphi/I");
  myTree->Branch("mom0_MB",  &mom0_MB,  "mom0_MB/F");
  myTree->Branch("mom1_MB",  &mom1_MB,  "mom1_MB/F");
  myTree->Branch("mom2_MB",  &mom2_MB,  "mom2_MB/F");
  myTree->Branch("mom3_MB",  &mom2_MB,  "mom3_MB/F");
  myTree->Branch("mom4_MB",  &mom4_MB,  "mom4_MB/F");
  myTree->Branch("trigbit",  &trigbit,  "trigbit/I");
  myTree->Branch("rnnumber", &rnnumber, "rnnumber/D");

  cells = 0;
  for (std::map<std::pair<int,HcalDetId>,HcalRecMinbias::myInfo>::const_iterator itr=count->myMap_.begin(); itr != count->myMap_.end(); ++itr) {
    edm::LogInfo("AnalyzerMB") << "Fired trigger bit number "<<itr->first.first;
    HcalRecMinbias::myInfo info = itr->second;
    if (info.theMB0 > 0) { 
      mom0_MB  = info.theMB0;
      mom1_MB  = info.theMB1;
      mom2_MB  = info.theMB2;
      mom3_MB  = info.theMB3;
      mom4_MB  = info.theMB4;
      rnnumber = info.runcheck;
      trigbit  = itr->first.first;
      mysubd   = itr->first.second.subdet();
      depth    = itr->first.second.depth();
      iphi     = itr->first.second.iphi();
      ieta     = itr->first.second.ieta();
      edm::LogInfo("AnalyzerMB") << " Result=  " << trigbit << " " << mysubd 
				 << " " << ieta << " " << iphi << " mom0  " 
				 << mom0_MB << " mom1 " << mom1_MB << " mom2 " 
				 << mom2_MB << " mom3 " << mom3_MB << " mom4 " 
				 << mom4_MB;
      myTree->Fill();
      cells++;
    }
  }
  edm::LogInfo("AnalyzerMB") << "cells" << " " << cells;
  
  hOutputFile->Write();   
  hOutputFile->cd();
//for (unsigned int i = 0; i<histo_.size(); i++) histo_[i]->Write();
  myTree->Write();
  hOutputFile->Close() ;
}

//
// member functions
//
// ------------ method called to produce the data  ------------
  
void RecAnalyzerMinbias::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    
  rnnum = (float)iEvent.run(); 
    
  edm::Handle<HBHERecHitCollection> hbheMB;
  iEvent.getByToken(tok_hbherecoMB_, hbheMB);
  if (!hbheMB.isValid()) {
    edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe product!";
    return ;
  }
  const HBHERecHitCollection HithbheMB = *(hbheMB.product());
  edm::LogInfo("AnalyzerMB") << "HBHE MB size of collection "<<HithbheMB.size();
  if (HithbheMB.size() < 5100 && runNZS_) {
    edm::LogWarning("AnalyzerMB") << "HBHE problem " << rnnum << " size "
				  << HithbheMB.size();
    return;
  }
    
  edm::Handle<HFRecHitCollection> hfMB;
  iEvent.getByToken(tok_hfrecoMB_, hfMB);
  if (!hfMB.isValid()) {
    edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe product!";
    return ;
  }
  const HFRecHitCollection HithfMB = *(hfMB.product());
  edm::LogInfo("AnalyzerMB") << "HF MB size of collection " << HithfMB.size();
  if (HithfMB.size() < 1700 && runNZS_) {
    edm::LogWarning("AnalyzerMB") << "HF problem " << rnnum << " size "
				  << HithfMB.size();
    return;
  }

  if (ignoreL1_) {
    analyzeHcal(HithbheMB, HithfMB, 1);
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
	  analyzeHcal(HithbheMB, HithfMB, algoBit);
	  ok          = true;
	} 
      }
      if (!ok) {
	edm::LogInfo("AnalyzerMB") << "No passed L1 Trigger found";
      }
    }
  }
}

void RecAnalyzerMinbias::analyzeHcal(const HBHERecHitCollection & HithbheMB,
				     const HFRecHitCollection & HithfMB,
				     int algoBit) {
  // Signal part for HB HE
  for (HBHERecHitCollection::const_iterator hbheItr=HithbheMB.begin(); 
       hbheItr!=HithbheMB.end(); hbheItr++) {
    // Recalibration of energy
    DetId mydetid = hbheItr->id().rawId();
    double icalconst(1.);	 
    if (theRecalib_) {
      std::map<DetId,double>::iterator itr = corrFactor_.find(mydetid);
      if (itr != corrFactor_.end()) icalconst = itr->second;
    }
    HBHERecHit aHit(hbheItr->id(),hbheItr->energy()*icalconst,hbheItr->time());
    double energyhit = aHit.energy();
    DetId id         = (*hbheItr).detid(); 
    HcalDetId hid    = HcalDetId(id);
    double eLow      = (hid.subdet() == HcalEndcap) ? eLowHE_  : eLowHB_;
    double eHigh     = (hid.subdet() == HcalEndcap) ? eHighHE_ : eHighHB_;
    /*
    for (unsigned int i = 0; i < hcalID_.size(); i++) {
      if (hcalID_[i] == id.rawId()) {
	histo_[i]->Fill(energyhit);
	break;
      }
    }
    */
    if (runNZS_ || (energyhit >= eLow && energyhit <= eHigh)) {
      std::map<std::pair<int,HcalDetId>,HcalRecMinbias::myInfo>::iterator itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
      if (itr1 == myMap_.end()) {
	HcalRecMinbias::myInfo info;
	myMap_[std::pair<int,HcalDetId>(algoBit,hid)] = info;
	itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
      } 
      itr1->second.theMB0++;
      itr1->second.theMB1 += energyhit;
      itr1->second.theMB2 += (energyhit*energyhit);
      itr1->second.theMB3 += (energyhit*energyhit*energyhit);
      itr1->second.theMB4 += (energyhit*energyhit*energyhit*energyhit);
      itr1->second.runcheck = rnnum;
    }
  } // HBHE_MB
 
  // Signal part for HF
  for (HFRecHitCollection::const_iterator hfItr=HithfMB.begin(); 
       hfItr!=HithfMB.end(); hfItr++) {
    // Recalibration of energy
    DetId mydetid = hfItr->id().rawId();
    double icalconst(1.);	 
    if (theRecalib_) {
      std::map<DetId,double>::iterator itr = corrFactor_.find(mydetid);
      if (itr != corrFactor_.end()) icalconst = itr->second;
    }
    HFRecHit aHit(hfItr->id(),hfItr->energy()*icalconst,hfItr->time());
    
    double energyhit = aHit.energy();
    DetId id         = (*hfItr).detid(); 
    HcalDetId hid    = HcalDetId(id);
    /*
    for (unsigned int i = 0; i < hcalID_.size(); i++) {
      if (hcalID_[i] == id.rawId()) {
	histo_[i]->Fill(energyhit);
	break;
      }
    }
    */
    //
    // Remove PMT hits
    //	 
    if ((runNZS_ && fabs(energyhit) <= 40.) || 
	(energyhit >= eLowHF_ && energyhit <= eHighHF_)) {
      std::map<std::pair<int,HcalDetId>,HcalRecMinbias::myInfo>::iterator itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
      if (itr1 == myMap_.end()) {
	HcalRecMinbias::myInfo info;
	myMap_[std::pair<int,HcalDetId>(algoBit,hid)] = info;
	itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
      }
      itr1->second.theMB0++;
      itr1->second.theMB1 += energyhit;
      itr1->second.theMB2 += (energyhit*energyhit);
      itr1->second.theMB3 += (energyhit*energyhit*energyhit);
      itr1->second.theMB4 += (energyhit*energyhit*energyhit*energyhit);
      itr1->second.runcheck = rnnum;
    }
  }
}

//define this as a plug-in                                                      
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecAnalyzerMinbias);
