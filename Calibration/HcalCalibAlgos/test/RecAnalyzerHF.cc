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
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "TH1D.h"
#include "TTree.h"

//#define EDM_ML_DEBUG

// class declaration
class RecAnalyzerHF : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit RecAnalyzerHF(const edm::ParameterSet&);
  ~RecAnalyzerHF();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void beginJob() override;
  virtual void endJob() override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
    
private:
  void analyzeHcal(const HFPreRecHitCollection&, int, bool);

  // ----------member data ---------------------------
  edm::Service<TFileService> fs_;
  bool                                ignoreL1_, nzs_, noise_, ratio_,fillTree_;
  double                              eLowHF_, eHighHF_;
  std::vector<unsigned int>           hcalID_;
  TTree                              *myTree_;
  TH1D                               *hist_[2];
  std::vector<std::pair<TH1D*,TH1D*>> histo_;
  std::vector<int>                    trigbit_;
  double                              rnnum_;
  struct myInfo{
    double kount, f11, f12, f13, f14, f21, f22, f23, f24, runcheck;
    myInfo() {
      kount = f11 = f12 = f13 = f14 = f21 = f22 = f23 = f24 = runcheck = 0;
    }
  };
  // Root tree members
  double                     rnnumber;
  int                        mysubd, depth, iphi, ieta, cells, trigbit;
  float                      mom0_F1, mom1_F1, mom2_F1, mom3_F1, mom4_F1;
  float                      mom0_F2, mom1_F2, mom2_F2, mom3_F2, mom4_F2;
  std::map<std::pair<int,HcalDetId>,myInfo>        myMap_;
  edm::EDGetTokenT<HFPreRecHitCollection>          tok_hfreco_;
  edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord> tok_hltL1GtMap_;
};

// constructors and destructor
  RecAnalyzerHF::RecAnalyzerHF(const edm::ParameterSet& iConfig) {

  usesResource("TFileService");
  // get name of output file with histogramms
  nzs_                  = iConfig.getParameter<bool>("RunNZS");
  noise_                = iConfig.getParameter<bool>("Noise");
  ratio_                = iConfig.getParameter<bool>("Ratio");
  eLowHF_               = iConfig.getParameter<double>("ELowHF");
  eHighHF_              = iConfig.getParameter<double>("EHighHF");
  trigbit_              = iConfig.getUntrackedParameter<std::vector<int>>("TriggerBits");
  ignoreL1_             = iConfig.getUntrackedParameter<bool>("IgnoreL1",false);
  fillTree_             = iConfig.getUntrackedParameter<bool>("FillTree",true);
  std::vector<int> ieta = iConfig.getUntrackedParameter<std::vector<int>>("HcalIeta");
  std::vector<int> iphi = iConfig.getUntrackedParameter<std::vector<int>>("HcalIphi");
  std::vector<int> depth= iConfig.getUntrackedParameter<std::vector<int>>("HcalDepth");

  // get token names of modules, producing object collections
  tok_hfreco_       = consumes<HFPreRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInput"));
  tok_hltL1GtMap_   = consumes<L1GlobalTriggerObjectMapRecord>(edm::InputTag("hltL1GtObjectMap"));

  edm::LogInfo("AnalyzerHF") << " Flags (IgnoreL1): " << ignoreL1_ << " (NZS) " 
			     << nzs_ << " (Noise) " << noise_ << " (Ratio) "
			     << ratio_;
  edm::LogInfo("AnalyzerHF") << "Thresholds for HF " << eLowHF_ << ":" 
			     << eHighHF_;
  for (unsigned int k=0; k<ieta.size(); ++k) {
    if (std::abs(ieta[k]) >= 29) {
      unsigned int id = (HcalDetId(HcalForward,ieta[k],iphi[k],depth[k])).rawId();
      hcalID_.push_back(id);
      edm::LogInfo("AnalyzerHF") << "DetId[" << k << "] " << HcalDetId(id);
    }
  }
  edm::LogInfo("AnalyzerHF") << "Select on " << trigbit_.size() 
			     << " L1 Trigger selection";
  unsigned int k(0);
  for (auto trig : trigbit_) {
    edm::LogInfo("AnalyzerHF") << "Bit[" << k << "] " << trig; ++k;}
}
  
RecAnalyzerHF::~RecAnalyzerHF() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void RecAnalyzerHF::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<bool>("RunNZS",true);
  desc.add<bool>("Noise",false);
  desc.add<bool>("Ratio",false);
  desc.add<double>("ELowHF",10);
  desc.add<double>("EHighHF",150);
  std::vector<int> idummy;
  desc.addUntracked<std::vector<int> >("TriggerBits",idummy);
  desc.addUntracked<bool>("IgnoreL1",false);
  desc.addUntracked<bool>("FillHisto",false);
  desc.addUntracked<std::vector<int> >("HcalIeta",idummy);
  desc.addUntracked<std::vector<int> >("HcalIphi",idummy);
  desc.addUntracked<std::vector<int> >("HcalDepth",idummy);
  desc.add<edm::InputTag>("hfInput",edm::InputTag("hfprereco"));
  descriptions.add("recAnalyzerHF",desc);
}
  
void RecAnalyzerHF::beginJob() {

  char        name[700], title[700];
  double      xmin(-10.0), xmax(90.0);
  if (ratio_) { xmin = -5.0; xmax = 5.0;}
  for (int i=0; i<2; ++i) {
    sprintf(name, "HF%d", i);
    sprintf (title, "The metric F%d for HF", i+1);
    hist_[i] = fs_->make<TH1D>(name,title,50,xmin,xmax);
  }

  for (auto id : hcalID_) {
    HcalDetId hid = HcalDetId(id);
    TH1D *h1(0), *h2(0);
    for (int i=0; i<2; ++i) {
      sprintf (name, "HF%d%d_%d_%d", i, hid.ieta(), hid.iphi(), hid.depth());
      sprintf (title, "The metric F%d for HF i#eta %d i#phi %d depth %d", 
	       i+1, hid.ieta(), hid.iphi(), hid.depth());
      if (i == 0) h1 = fs_->make<TH1D>(name, title, 50, xmin, xmax);
      else        h2 = fs_->make<TH1D>(name, title, 50, xmin, xmax);
    }
    histo_.push_back(std::pair<TH1D*,TH1D*>(h1,h2));
  };

  if (fillTree_) {
    myTree_       = fs_->make<TTree>("RecJet","RecJet Tree");
    myTree_->Branch("cells",    &cells,    "cells/I");
    myTree_->Branch("mysubd",   &mysubd,   "mysubd/I");
    myTree_->Branch("depth",    &depth,    "depth/I");
    myTree_->Branch("ieta",     &ieta,     "ieta/I");
    myTree_->Branch("iphi",     &iphi,     "iphi/I");
    myTree_->Branch("mom0_F1",  &mom0_F1,  "mom0_F1/F");
    myTree_->Branch("mom1_F1",  &mom1_F1,  "mom1_F1/F");
    myTree_->Branch("mom2_F1",  &mom2_F1,  "mom2_F1/F");
    myTree_->Branch("mom3_F1",  &mom3_F1,  "mom3_F1/F");
    myTree_->Branch("mom4_F1",  &mom4_F1,  "mom4_F1/F");
    myTree_->Branch("mom0_F2",  &mom0_F2,  "mom0_F2/F");
    myTree_->Branch("mom1_F2",  &mom1_F2,  "mom1_F2/F");
    myTree_->Branch("mom2_F2",  &mom2_F2,  "mom2_F2/F");
    myTree_->Branch("mom3_F2",  &mom3_F2,  "mom3_F2/F");
    myTree_->Branch("mom4_F2",  &mom4_F2,  "mom4_F2/F");
    myTree_->Branch("trigbit",  &trigbit,  "trigbit/I");
    myTree_->Branch("rnnumber", &rnnumber, "rnnumber/D");
  }

  myMap_.clear();
}

void RecAnalyzerHF::endJob() {

  if (fillTree_) {
    cells = 0;
    for (auto itr : myMap_) {
      edm::LogInfo("AnalyzerHF") << "Fired trigger bit number "<<itr.first.first;
#ifdef EDM_ML_DEBUG
      std::cout << "Fired trigger bit number "<<itr->first.first << std::endl;
#endif
      myInfo info = itr.second;
      if (info.kount > 0) { 
	mom0_F1  = info.kount;
	mom1_F1  = info.f11;
	mom2_F1  = info.f12;
	mom3_F1  = info.f13;
	mom4_F1  = info.f14;
	mom0_F2  = info.kount;
	mom1_F2  = info.f21;
	mom2_F2  = info.f22;
	mom3_F2  = info.f23;
	mom4_F2  = info.f24;
	rnnumber = info.runcheck;
	trigbit  = itr.first.first;
	mysubd   = itr.first.second.subdet();
	depth    = itr.first.second.depth();
	iphi     = itr.first.second.iphi();
	ieta     = itr.first.second.ieta();
	edm::LogInfo("AnalyzerHF") << " Result=  " << trigbit << " " << mysubd 
				   << " " << ieta << " " << iphi 
				   << " F1:mom0  " << mom0_F1 << " mom1 " 
				   << mom1_F1 << " mom2 " << mom2_F1 << " mom3 "
				   << mom3_F1 << " mom4 " << mom4_F1
				   << " F2:mom0 " << mom0_F2  << " mom1 " 
				   << mom1_F2 << " mom2 " << mom2_F2 << " mom3 "
				   << mom3_F2 << " mom4 " << mom4_F2;
#ifdef EDM_ML_DEBUG
	std::cout << " Result=  " << trigbit << " " << mysubd 
		  << " " << ieta << " " << iphi 
		  << " F1:mom0  " << mom0_F1 << " mom1 " 
		  << mom1_F1 << " mom2 " << mom2_F1 << " mom3 "
		  << mom3_F1 << " mom4 " << mom4_F1
		  << " F2:mom0 " << mom0_F2  << " mom1 " 
		  << mom1_F2 << " mom2 " << mom2_F2 << " mom3 "
		  << mom3_F2 << " mom4 " << mom4_F2 << std::endl;
#endif
	myTree_->Fill();
	cells++;
      }
    }
    edm::LogInfo("AnalyzerHF") << "cells" << " " << cells;
#ifdef EDM_ML_DEBUG
    std::cout << "cells" << " " << cells << std::endl;
#endif
  }
#ifdef EDM_ML_DEBUG
  std::cout << "Exiting from RecAnalyzerHF::endjob" << std::endl;
#endif
}

//
// member functions
//
// ------------ method called to produce the data  ------------
  
void RecAnalyzerHF::analyze(const edm::Event& iEvent, const edm::EventSetup&) {

  rnnum_ = (double)iEvent.run(); 
    
  edm::Handle<HFPreRecHitCollection> hf;
  iEvent.getByToken(tok_hfreco_, hf);
  if (!hf.isValid()) {
    edm::LogWarning("AnalyzerHF") << "HcalCalibAlgos: Error! can't get hf product!";
    return;
  }
  const HFPreRecHitCollection Hithf = *(hf.product());
  edm::LogInfo("AnalyzerHF") << "HF MB size of collection " << Hithf.size();
  if (Hithf.size() < 1700 && nzs_) {
    edm::LogWarning("AnalyzerHF") << "HF problem " << rnnum_ << " size "
				  << Hithf.size();
  }

  bool select(false);
  if (trigbit_.size() > 0) {
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByToken(tok_hltL1GtMap_, gtObjectMapRecord);
    if (gtObjectMapRecord.isValid()) {
      const std::vector<L1GlobalTriggerObjectMap>& objMapVec = gtObjectMapRecord->gtObjectMap();
      for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
           itMap != objMapVec.end(); ++itMap) {
        bool resultGt = (*itMap).algoGtlResult();
        if (resultGt) {
          int algoBit = (*itMap).algoBitNumber();
          if (std::find(trigbit_.begin(),trigbit_.end(),algoBit) != 
	      trigbit_.end()) {
            select = true;
            break;
          }
        }
      }
    }
  }

  //event weight for FLAT sample and PU information

  if (ignoreL1_ || ((trigbit_.size() > 0) && select)) {
    analyzeHcal(Hithf, 1, true);
  } else if ((!ignoreL1_) && (trigbit_.size() == 0)) {
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
	  analyzeHcal(Hithf, algoBit, (!ok));
	  ok          = true;
	} 
      }
      if (!ok) {
	edm::LogInfo("AnalyzerHF") << "No passed L1 Trigger found";
      }
    }
  }
}

void RecAnalyzerHF::analyzeHcal(const HFPreRecHitCollection & Hithf,
				int algoBit, bool fill) {
 
  // Signal part for HF
  for (auto hfItr : Hithf ) {
    double e0     = hfItr.getHFQIE10Info(0)->energy();
    double e1     = hfItr.getHFQIE10Info(1)->energy();
    double energy = e0+e1;
    if (std::abs(energy) < 1e-6) energy = (energy > 0) ? 1e-6 : -1e-6;
    double f1(e0), f2(e1);
    if (ratio_) {
      f1         /= energy;
      f2         /= energy;
    }
    HcalDetId hid = hfItr.id(); 
    if (fill) {
      for (unsigned int i = 0; i < hcalID_.size(); i++) {
	if (hcalID_[i] == hid.rawId()) {
	  histo_[i].first->Fill(f1);
	  histo_[i].second->Fill(f2);
	  break;
	}
      }
      hist_[0]->Fill(f1);
      hist_[1]->Fill(f2);
    }

    //
    // Remove PMT hits
    //
    if (((noise_ || nzs_) && fabs(energy) <= 40.) || 
	(energy >= eLowHF_ && energy <= eHighHF_)) {
      std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
      if (itr1 == myMap_.end()) {
	myInfo info;
	myMap_[std::pair<int,HcalDetId>(algoBit,hid)] = info;
	itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
      }
      itr1->second.kount   += 1.0;
      itr1->second.f11     += (f1);
      itr1->second.f12     += (f1*f1);
      itr1->second.f13     += (f1*f1*f1);
      itr1->second.f14     += (f1*f1*f1*f1);
      itr1->second.f21     += (f2);
      itr1->second.f22     += (f2*f2);
      itr1->second.f23     += (f2*f2*f2);
      itr1->second.f24     += (f2*f2*f2*f2);
      itr1->second.runcheck = rnnum_;
    }
  }
}

//define this as a plug-in                                                      
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecAnalyzerHF);
