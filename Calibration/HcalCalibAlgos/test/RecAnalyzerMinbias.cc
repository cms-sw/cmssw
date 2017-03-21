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
#include "TFile.h"
#include "TTree.h"
#include "TMath.h"
#include "TF1.h"

//#define DebugLog

// class declaration
class RecAnalyzerMinbias : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit RecAnalyzerMinbias(const edm::ParameterSet&);
  ~RecAnalyzerMinbias();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void beginJob() override;
  virtual void endJob() override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
    
private:
  void analyzeHcal(const HBHERecHitCollection&, const HFRecHitCollection&, int, bool, double);

  // ----------member data ---------------------------
  edm::Service<TFileService> fs_;
  bool                       theRecalib_, ignoreL1_, runNZS_, Noise_, fillHist_, init_;
  double                     eLowHB_, eHighHB_, eLowHE_, eHighHE_;
  double                     eLowHF_, eHighHF_;
  std::map<DetId,double>     corrFactor_;
  std::vector<unsigned int>  hcalID_;
  TTree                     *myTree_, *myTree1_;
  TH1D                      *h_[4];
  std::vector<TH1D*>         histo_;
  std::map<HcalDetId,TH1D*>  histHC_;
  std::vector<int>           trigbit_;
  double                     rnnum_;
  struct myInfo{
    double theMB0, theMB1, theMB2, theMB3, theMB4, runcheck;
    myInfo() {
      theMB0 = theMB1 = theMB2 = theMB3 = theMB4 = runcheck = 0;
    }
  };
  // Root tree members
  double                     rnnumber;
  int                        mysubd, depth, iphi, ieta, cells, trigbit;
  float                      mom0_MB, mom1_MB, mom2_MB, mom3_MB, mom4_MB;
  int                        HBHEsize, HFsize;
  std::map<std::pair<int,HcalDetId>,myInfo>        myMap_;
  edm::EDGetTokenT<HBHERecHitCollection>           tok_hbherecoMB_;
  edm::EDGetTokenT<HFRecHitCollection>             tok_hfrecoMB_;
  edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord> tok_hltL1GtMap_;
  edm::EDGetTokenT<GenEventInfoProduct>            tok_ew_; 
};

// constructors and destructor
RecAnalyzerMinbias::RecAnalyzerMinbias(const edm::ParameterSet& iConfig) :
  init_(false) {

  usesResource("TFileService");

  // get name of output file with histogramms
  runNZS_               = iConfig.getParameter<bool>("RunNZS");
  Noise_                = iConfig.getParameter<bool>("Noise");
  eLowHB_               = iConfig.getParameter<double>("ELowHB");
  eHighHB_              = iConfig.getParameter<double>("EHighHB");
  eLowHE_               = iConfig.getParameter<double>("ELowHE");
  eHighHE_              = iConfig.getParameter<double>("EHighHE");
  eLowHF_               = iConfig.getParameter<double>("ELowHF");
  eHighHF_              = iConfig.getParameter<double>("EHighHF");
  trigbit_              = iConfig.getUntrackedParameter<std::vector<int>>("TriggerBits");
  ignoreL1_             = iConfig.getUntrackedParameter<bool>("IgnoreL1",false);
  std::string      cfile= iConfig.getUntrackedParameter<std::string>("CorrFile");
  fillHist_             = iConfig.getUntrackedParameter<bool>("FillHisto",false);
  std::vector<int> ieta = iConfig.getUntrackedParameter<std::vector<int>>("HcalIeta");
  std::vector<int> iphi = iConfig.getUntrackedParameter<std::vector<int>>("HcalIphi");
  std::vector<int> depth= iConfig.getUntrackedParameter<std::vector<int>>("HcalDepth");

  // get token names of modules, producing object collections
  tok_hbherecoMB_   = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputMB"));
  tok_hfrecoMB_     = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputMB"));
  tok_hltL1GtMap_   = consumes<L1GlobalTriggerObjectMapRecord>(edm::InputTag("hltL1GtObjectMap"));
  tok_ew_           = consumes<GenEventInfoProduct>(edm::InputTag("generator"));

  // Read correction factors
  std::ifstream infile(cfile.c_str());
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
  edm::LogInfo("AnalyzerMB") << "Select on " << trigbit_.size() 
			     << " L1 Trigger selection";
  for (unsigned int k=0; k<trigbit_.size(); ++k)
    edm::LogInfo("AnalyzerMB") << "Bit[" << k << "] " << trigbit_[k];
}
  
RecAnalyzerMinbias::~RecAnalyzerMinbias() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void RecAnalyzerMinbias::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
  
void RecAnalyzerMinbias::beginJob() {

  std::string hc[5] = {"Empty", "HB", "HE", "HO", "HF"};
  char        name[700], title[700];
  for(int idet=1; idet<=4; idet++){
    sprintf(name, "%s", hc[idet].c_str());
    sprintf (title, "Noise distribution for %s", hc[idet].c_str());
    h_[idet-1] = fs_->make<TH1D>(name,title,48,-6., 6.);
  }

  for (unsigned int i=0; i<hcalID_.size(); i++) {
    HcalDetId id = HcalDetId(hcalID_[i]);
    int subdet   = id.subdetId();
    sprintf (name, "%s%d_%d_%d", hc[subdet].c_str(), id.ieta(), id.iphi(), id.depth());
    sprintf (title, "Energy Distribution for %s ieta %d iphi %d depth %d", hc[subdet].c_str(), id.ieta(), id.iphi(), id.depth());
    double xmin = (subdet == 4) ? -10 : -1;
    double xmax = (subdet == 4) ? 90 : 9;
    TH1D*  hh   = fs_->make<TH1D>(name, title, 50, xmin, xmax);
    histo_.push_back(hh);
  };

  if (!fillHist_) {
    myTree_       = fs_->make<TTree>("RecJet","RecJet Tree");
    myTree_->Branch("cells",    &cells,    "cells/I");
    myTree_->Branch("mysubd",   &mysubd,   "mysubd/I");
    myTree_->Branch("depth",    &depth,    "depth/I");
    myTree_->Branch("ieta",     &ieta,     "ieta/I");
    myTree_->Branch("iphi",     &iphi,     "iphi/I");
    myTree_->Branch("mom0_MB",  &mom0_MB,  "mom0_MB/F");
    myTree_->Branch("mom1_MB",  &mom1_MB,  "mom1_MB/F");
    myTree_->Branch("mom2_MB",  &mom2_MB,  "mom2_MB/F");
    myTree_->Branch("mom3_MB",  &mom3_MB,  "mom3_MB/F");
    myTree_->Branch("mom4_MB",  &mom4_MB,  "mom4_MB/F");
    myTree_->Branch("trigbit",  &trigbit,  "trigbit/I");
    myTree_->Branch("rnnumber", &rnnumber, "rnnumber/D");
  }
  myTree1_      = fs_->make<TTree>("RecJet1","RecJet1 Tree");
  myTree1_->Branch("rnnum_",   &rnnum_,    "rnnum_/D");
  myTree1_->Branch("HBHEsize", &HBHEsize,  "HBHEsize/I");
  myTree1_->Branch("HFsize",   &HFsize,    "HFsize/I");

  myMap_.clear();
}

//  EndJob
//
void RecAnalyzerMinbias::endJob() {

  if (!fillHist_) {
    cells = 0;
    for (std::map<std::pair<int,HcalDetId>,myInfo>::const_iterator itr=myMap_.begin(); itr != myMap_.end(); ++itr) {
      edm::LogInfo("AnalyzerMB") << "Fired trigger bit number "<<itr->first.first;
#ifdef DebugLog
      std::cout << "Fired trigger bit number "<<itr->first.first << std::endl;
#endif
      myInfo info = itr->second;
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
#ifdef DebugLog
	std::cout << " Result=  " << trigbit << " " << mysubd 
		  << " " << ieta << " " << iphi << " mom0  " 
		  << mom0_MB << " mom1 " << mom1_MB << " mom2 " 
		  << mom2_MB << " mom3 " << mom3_MB << " mom4 " 
		  << mom4_MB << std::endl;
#endif
	myTree_->Fill();
	cells++;
      }
    }
    edm::LogInfo("AnalyzerMB") << "cells" << " " << cells;
#ifdef DebugLog
    std::cout << "cells" << " " << cells << std::endl;
#endif
  }
#ifdef DebugLog
  std::cout << "Exiting from RecAnalyzerMinbias::endjob" << std::endl;
#endif
}
  
void RecAnalyzerMinbias::beginRun(edm::Run const&, edm::EventSetup const& iS) {
 
  if (!init_) {
    init_ = true;
    if (fillHist_) {
      edm::ESHandle<HcalTopology> htopo;
      iS.get<IdealGeometryRecord>().get(htopo);
      if (htopo.isValid()) {
	const HcalTopology* hcaltopology = htopo.product();

	char  name[700], title[700];
	// For HB
	int maxDepthHB = hcaltopology->maxDepthHB();
	int nbinHB     = (Noise_) ? 18 : int(2000*eHighHB_);
	double x_min   = (Noise_) ? -3.   : 0.;
	double x_max   = (Noise_) ? 3.    :  2.*eHighHB_;
	for (int eta = -50; eta < 50; eta++) {
	  for (int phi = 0; phi < 100; phi++) {
	    for (int depth = 1; depth <= maxDepthHB; depth++) {
	      HcalDetId cell (HcalBarrel, eta, phi, depth);
	      if (hcaltopology->valid(cell)) {
		sprintf (name, "HBeta%dphi%ddep%d", eta, phi, depth);
		sprintf (title,"HB #eta %d #phi %d depth %d", eta, phi, depth);
		TH1D* h = fs_->make<TH1D>(name, title, nbinHB, x_min, x_max);  
		histHC_[cell] = h;
	      }
	    }
	  }
	}
	// For HE
	int maxDepthHE = hcaltopology->maxDepthHE();
	int nbinHE     = (Noise_) ? 18 : int(2000*eHighHE_);
        x_min   = (Noise_) ? -3.   : 0.;
        x_max   = (Noise_) ? 3.    :  2.*eHighHE_;
	for (int eta = -50; eta < 50; eta++) {
	  for (int phi = 0; phi < 100; phi++) {
	    for (int depth = 1; depth <= maxDepthHE; depth++) {
	      HcalDetId cell (HcalEndcap, eta, phi, depth);
	      if (hcaltopology->valid(cell)) {
		sprintf (name, "HEeta%dphi%ddep%d", eta, phi, depth);
		sprintf (title,"HE #eta %d #phi %d depth %d", eta, phi, depth);
		TH1D* h = fs_->make<TH1D>(name, title, nbinHE, x_min, x_max);
		histHC_[cell] = h;
	      }
	    }
	  }
	}
	// For HF
	int maxDepthHF = 4;
	int nbinHF     = (Noise_) ? 200 : int(2000*eHighHF_);
	x_min   = (Noise_) ? -10.   : 0.;
	x_max   = (Noise_) ? 10.    :  2.*eHighHF_;
	for (int eta = -50; eta < 50; eta++) {
	  for (int phi = 0; phi < 100; phi++) {
	    for (int depth = 1; depth <= maxDepthHF; depth++) {
	      HcalDetId cell (HcalForward, eta, phi, depth);
	      if (hcaltopology->valid(cell)) {
		sprintf (name, "HFeta%dphi%ddep%d", eta, phi, depth);
		sprintf (title,"Energy (HF #eta %d #phi %d depth %d)", eta, phi, depth);
		TH1D* h = fs_->make<TH1D>(name, title, nbinHF, x_min, x_max);
		histHC_[cell] = h;
	      }
	    }
	  }
	}
      }
    }
  }
}

//
// member functions
//
// ------------ method called to produce the data  ------------
  
void RecAnalyzerMinbias::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  rnnum_ = (double)iEvent.run(); 
  edm::Handle<HBHERecHitCollection> hbheMB;
  iEvent.getByToken(tok_hbherecoMB_, hbheMB);
  if (!hbheMB.isValid()) {
    edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe product!";
    return ;
  }
  const HBHERecHitCollection HithbheMB = *(hbheMB.product());
  HBHEsize = HithbheMB.size();
  edm::LogInfo("AnalyzerMB") << "HBHE MB size of collection "<<HithbheMB.size();
  if (HithbheMB.size() < 5100 && runNZS_) {
    edm::LogWarning("AnalyzerMB") << "HBHE problem " << rnnum_ << " size "
				  << HBHEsize;
  }
    
  edm::Handle<HFRecHitCollection> hfMB;
  iEvent.getByToken(tok_hfrecoMB_, hfMB);
  if (!hfMB.isValid()) {
    edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hf product!";
    return;
  }
  const HFRecHitCollection HithfMB = *(hfMB.product());
  edm::LogInfo("AnalyzerMB") << "HF MB size of collection " << HithfMB.size();
  HFsize = HithfMB.size();
  if (HithfMB.size() < 1700 && runNZS_) {
    edm::LogWarning("AnalyzerMB") << "HF problem " << rnnum_ << " size "
				  << HFsize;
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

  if ((trigbit_.size() == 0) || select) myTree1_->Fill();

  //event weight for FLAT sample and PU information
  double eventWeight = 1.0;
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(tok_ew_, genEventInfo);
  if (genEventInfo.isValid()) eventWeight = genEventInfo->weight();  

  if (ignoreL1_ || ((trigbit_.size() > 0) && select)) {
    analyzeHcal(HithbheMB, HithfMB, 1, true, eventWeight);
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
	  analyzeHcal(HithbheMB, HithfMB, algoBit, (!ok), eventWeight);
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
				     int algoBit, bool fill, double weight) {
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
    if (fill) {
      for (unsigned int i = 0; i < hcalID_.size(); i++) {
	if (hcalID_[i] == id.rawId()) {
	  histo_[i]->Fill(energyhit);
	  break;
	}
      }
      if (fillHist_) {
	std::map<HcalDetId,TH1D*>::iterator itr1 = histHC_.find(hid);
	if (itr1 != histHC_.end()) itr1->second->Fill(energyhit);
      }
      h_[hid.subdet()-1]->Fill(energyhit);
    }
    if (!fillHist_) {
      if (Noise_ || runNZS_ || (energyhit >= eLow && energyhit <= eHigh)) {
	std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
	if (itr1 == myMap_.end()) {
	  myInfo info;
	  myMap_[std::pair<int,HcalDetId>(algoBit,hid)] = info;
	  itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
	} 
	itr1->second.theMB0 += weight;
	itr1->second.theMB1 += (weight*energyhit);
	itr1->second.theMB2 += (weight*energyhit*energyhit);
	itr1->second.theMB3 += (weight*energyhit*energyhit*energyhit);
	itr1->second.theMB4 += (weight*energyhit*energyhit*energyhit*energyhit);
	itr1->second.runcheck = rnnum_;
      }
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
    if (fill) {
      for (unsigned int i = 0; i < hcalID_.size(); i++) {
	if (hcalID_[i] == id.rawId()) {
	  histo_[i]->Fill(energyhit);
	  break;
	}
      }
      if (fillHist_) {
	std::map<HcalDetId,TH1D*>::iterator itr1 = histHC_.find(hid);
	if (itr1 != histHC_.end()) itr1->second->Fill(energyhit);
      }
      h_[hid.subdet()-1]->Fill(energyhit);
    }

    //
    // Remove PMT hits
    //
    if (!fillHist_) {
      if (((Noise_ || runNZS_) && fabs(energyhit) <= 40.) || 
	  (energyhit >= eLowHF_ && energyhit <= eHighHF_)) {
	std::map<std::pair<int,HcalDetId>,myInfo>::iterator itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
	if (itr1 == myMap_.end()) {
	  myInfo info;
	  myMap_[std::pair<int,HcalDetId>(algoBit,hid)] = info;
	  itr1 = myMap_.find(std::pair<int,HcalDetId>(algoBit,hid));
	}
	itr1->second.theMB0 += weight;
	itr1->second.theMB1 += (weight*energyhit);
	itr1->second.theMB2 += (weight*energyhit*energyhit);
	itr1->second.theMB3 += (weight*energyhit*energyhit*energyhit);
	itr1->second.theMB4 += (weight*energyhit*energyhit*energyhit*energyhit);
	itr1->second.runcheck = rnnum_;
      }
    }
  }
}

//define this as a plug-in                                                      
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecAnalyzerMinbias);
