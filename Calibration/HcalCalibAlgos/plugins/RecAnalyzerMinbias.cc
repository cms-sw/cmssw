// -*- C++ -*-
//
// Package:    HcalCalibAlgos
// Class:      RecAnalyzerMinbias
//
/**\class RecAnalyzerMinbias RecAnalyzerMinbias.cc Calibration/HcalCalibAlgos/test/RecAnalyzerMinbias.cc

 Description: Performs phi-symmetry studies of HB/HE/HF channels

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Thu Mar  4 18:52:02 CST 2012
//
//

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
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TMath.h"
#include "TProfile.h"
#include "TTree.h"

//#define EDM_ML_DEBUG

// class declaration
class RecAnalyzerMinbias : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit RecAnalyzerMinbias(const edm::ParameterSet&);
  ~RecAnalyzerMinbias() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override;
  void endJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  void analyzeHcal(const HBHERecHitCollection&, const HFRecHitCollection&, int, bool, double);

  // ----------member data ---------------------------
  edm::Service<TFileService> fs_;
  bool theRecalib_, ignoreL1_, runNZS_, Noise_;
  bool fillHist_, extraHist_, init_;
  double eLowHB_, eHighHB_, eLowHE_, eHighHE_;
  double eLowHF_, eHighHF_, eMin_;
  int runMin_, runMax_;
  std::map<DetId, double> corrFactor_;
  std::vector<unsigned int> hcalID_;
  TTree *myTree_, *myTree1_;
  TH1D* h_[4];
  TH2D *hbhe_, *hb_, *he_, *hf_;
  TH1D *h_AmplitudeHBtest_, *h_AmplitudeHEtest_;
  TH1D* h_AmplitudeHFtest_;
  TH1D *h_AmplitudeHB_, *h_AmplitudeHE_, *h_AmplitudeHF_;
  TProfile *hbherun_, *hbrun_, *herun_, *hfrun_;
  std::vector<TH1D*> histo_;
  std::map<HcalDetId, TH1D*> histHC_;
  std::vector<int> trigbit_;
  double rnnum_;
  struct myInfo {
    double theMB0, theMB1, theMB2, theMB3, theMB4, runcheck;
    myInfo() { theMB0 = theMB1 = theMB2 = theMB3 = theMB4 = runcheck = 0; }
  };
  // Root tree members
  double rnnumber;
  int mysubd, depth, iphi, ieta, cells, trigbit;
  float mom0_MB, mom1_MB, mom2_MB, mom3_MB, mom4_MB;
  int HBHEsize, HFsize;
  std::map<std::pair<int, HcalDetId>, myInfo> myMap_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbherecoMB_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hfrecoMB_;
  edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord> tok_hltL1GtMap_;
  edm::EDGetTokenT<GenEventInfoProduct> tok_ew_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhedigi_;
  edm::EDGetTokenT<QIE11DigiCollection> tok_qie11digi_;
  edm::EDGetTokenT<HODigiCollection> tok_hodigi_;
  edm::EDGetTokenT<HFDigiCollection> tok_hfdigi_;
  edm::EDGetTokenT<QIE10DigiCollection> tok_qie10digi_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> tok_gtRec_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo_;
};

// constructors and destructor
RecAnalyzerMinbias::RecAnalyzerMinbias(const edm::ParameterSet& iConfig) : init_(false) {
  usesResource("TFileService");

  // get name of output file with histogramms
  runNZS_ = iConfig.getParameter<bool>("runNZS");
  Noise_ = iConfig.getParameter<bool>("noise");
  eLowHB_ = iConfig.getParameter<double>("eLowHB");
  eHighHB_ = iConfig.getParameter<double>("eHighHB");
  eLowHE_ = iConfig.getParameter<double>("eLowHE");
  eHighHE_ = iConfig.getParameter<double>("eHighHE");
  eLowHF_ = iConfig.getParameter<double>("eLowHF");
  eHighHF_ = iConfig.getParameter<double>("eHighHF");
  eMin_ = iConfig.getUntrackedParameter<double>("eMin", 2.0);
  // The following run range is suited to study 2017 commissioning period
  runMin_ = iConfig.getUntrackedParameter<int>("RunMin", 308327);
  runMax_ = iConfig.getUntrackedParameter<int>("RunMax", 315250);
  trigbit_ = iConfig.getUntrackedParameter<std::vector<int>>("triggerBits");
  ignoreL1_ = iConfig.getUntrackedParameter<bool>("ignoreL1", false);
  std::string cfile = iConfig.getUntrackedParameter<std::string>("corrFile");
  fillHist_ = iConfig.getUntrackedParameter<bool>("fillHisto", false);
  extraHist_ = iConfig.getUntrackedParameter<bool>("extraHisto", false);
  std::vector<int> ieta = iConfig.getUntrackedParameter<std::vector<int>>("hcalIeta");
  std::vector<int> iphi = iConfig.getUntrackedParameter<std::vector<int>>("hcalIphi");
  std::vector<int> depth = iConfig.getUntrackedParameter<std::vector<int>>("hcalDepth");

  // get token names of modules, producing object collections
  tok_hbherecoMB_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputMB"));
  tok_hfrecoMB_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputMB"));
  tok_hltL1GtMap_ = consumes<L1GlobalTriggerObjectMapRecord>(edm::InputTag("hltL1GtObjectMap"));
  tok_ew_ = consumes<GenEventInfoProduct>(edm::InputTag("generator"));
  tok_hbhedigi_ = consumes<HBHEDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalDigiCollectionTag"));
  tok_qie11digi_ = consumes<QIE11DigiCollection>(iConfig.getParameter<edm::InputTag>("hcalDigiCollectionTag"));
  tok_hodigi_ = consumes<HODigiCollection>(iConfig.getParameter<edm::InputTag>("hcalDigiCollectionTag"));
  tok_hfdigi_ = consumes<HFDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalDigiCollectionTag"));
  tok_qie10digi_ = consumes<QIE10DigiCollection>(iConfig.getParameter<edm::InputTag>("hcalDigiCollectionTag"));
  tok_gtRec_ = consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag("gtDigisAlCaMB"));

  tok_htopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord, edm::Transition::BeginRun>();

  // Read correction factors
  std::ifstream infile(cfile.c_str());
  if (!infile.is_open()) {
    theRecalib_ = false;
    edm::LogWarning("RecAnalyzer") << "Cannot open '" << cfile << "' for the correction file";
  } else {
    unsigned int ndets(0), nrec(0);
    while (true) {
      unsigned int id;
      double cfac;
      infile >> id >> cfac;
      if (!infile.good())
        break;
      HcalDetId detId(id);
      nrec++;
      std::map<DetId, double>::iterator itr = corrFactor_.find(detId);
      if (itr == corrFactor_.end()) {
        corrFactor_[detId] = cfac;
        ndets++;
      }
    }
    infile.close();
    edm::LogVerbatim("RecAnalyzer") << "Reads " << nrec << " correction factors for " << ndets << " detIds";
    theRecalib_ = (ndets > 0);
  }

  edm::LogVerbatim("RecAnalyzer") << " Flags (ReCalib): " << theRecalib_ << " (IgnoreL1): " << ignoreL1_ << " (NZS) "
                                  << runNZS_ << " and with " << ieta.size() << " detId for full histogram";
  edm::LogVerbatim("RecAnalyzer") << "Thresholds for HB " << eLowHB_ << ":" << eHighHB_ << "  for HE " << eLowHE_ << ":"
                                  << eHighHE_ << "  for HF " << eLowHF_ << ":" << eHighHF_;
  for (unsigned int k = 0; k < ieta.size(); ++k) {
    HcalSubdetector subd = ((std::abs(ieta[k]) > 29)                         ? HcalForward
                            : (std::abs(ieta[k]) > 16)                       ? HcalEndcap
                            : ((std::abs(ieta[k]) == 16) && (depth[k] == 3)) ? HcalEndcap
                            : (depth[k] == 4)                                ? HcalOuter
                                                                             : HcalBarrel);
    unsigned int id = (HcalDetId(subd, ieta[k], iphi[k], depth[k])).rawId();
    hcalID_.push_back(id);
    edm::LogVerbatim("RecAnalyzer") << "DetId[" << k << "] " << HcalDetId(id);
  }
  edm::LogVerbatim("RecAnalyzer") << "Select on " << trigbit_.size() << " L1 Trigger selection";
  for (unsigned int k = 0; k < trigbit_.size(); ++k)
    edm::LogVerbatim("RecAnalyzer") << "Bit[" << k << "] " << trigbit_[k];
}

RecAnalyzerMinbias::~RecAnalyzerMinbias() {}

void RecAnalyzerMinbias::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  std::vector<int> iarray;
  edm::ParameterSetDescription desc;
  desc.add<bool>("runNZS", true);
  desc.add<bool>("noise", false);
  desc.add<double>("eLowHB", 4);
  desc.add<double>("eHighHB", 100);
  desc.add<double>("eLowHE", 4);
  desc.add<double>("eHighHE", 150);
  desc.add<double>("eLowHF", 10);
  desc.add<double>("eHighHF", 150);
  // Suitable cutoff to remove fluctuation of pedestal
  desc.addUntracked<double>("eMin", 2.0);
  // The following run range is suited to study 2017 commissioning period
  desc.addUntracked<int>("runMin", 308327);
  desc.addUntracked<int>("runMax", 308347);
  desc.addUntracked<std::vector<int>>("triggerBits", iarray);
  desc.addUntracked<bool>("ignoreL1", false);
  desc.addUntracked<std::string>("corrFile", "CorFactor.txt");
  desc.addUntracked<bool>("fillHisto", false);
  desc.addUntracked<bool>("extraHisto", false);
  desc.addUntracked<std::vector<int>>("hcalIeta", iarray);
  desc.addUntracked<std::vector<int>>("hcalIphi", iarray);
  desc.addUntracked<std::vector<int>>("hcalDepth", iarray);
  desc.add<edm::InputTag>("hbheInputMB", edm::InputTag("hbherecoMB"));
  desc.add<edm::InputTag>("hfInputMB", edm::InputTag("hfrecoMB"));
  desc.add<edm::InputTag>("gtDigisAlCaMB", edm::InputTag("gtDigisAlCaMB"));
  desc.add<edm::InputTag>("hcalDigiCollectionTag", edm::InputTag("hcalDigis"));
  descriptions.add("recAnalyzerMinbias", desc);
}

void RecAnalyzerMinbias::beginJob() {
  std::string hc[5] = {"Empty", "HB", "HE", "HO", "HF"};
  char name[700], title[700];
  hbhe_ = fs_->make<TH2D>("hbhe", "Noise in HB/HE", 61, -30.5, 30.5, 72, 0.5, 72.5);
  hb_ = fs_->make<TH2D>("hb", "Noise in HB", 61, -16.5, 16.5, 72, 0.5, 72.5);
  he_ = fs_->make<TH2D>("he", "Noise in HE", 61, -30.5, 30.5, 72, 0.5, 72.5);
  hf_ = fs_->make<TH2D>("hf", "Noise in HF", 82, -41.5, 41.5, 72, 0.5, 72.5);
  int nbin = (runMax_ - runMin_ + 1);
  sprintf(title, "Fraction of channels in HB/HE with E > %4.1f GeV vs Run number", eMin_);
  hbherun_ = fs_->make<TProfile>("hbherun", title, nbin, runMin_ - 0.5, runMax_ + 0.5, 0.0, 1.0);
  sprintf(title, "Fraction of channels in HB with E > %4.1f GeV vs Run number", eMin_);
  hbrun_ = fs_->make<TProfile>("hbrun", title, nbin, runMin_ - 0.5, runMax_ + 0.5, 0.0, 1.0);
  sprintf(title, "Fraction of channels in HE with E > %4.1f GeV vs Run number", eMin_);
  herun_ = fs_->make<TProfile>("herun", title, nbin, runMin_ - 0.5, runMax_ + 0.5, 0.0, 1.0);
  sprintf(title, "Fraction of channels in HF with E > %4.1f GeV vs Run number", eMin_);
  hfrun_ = fs_->make<TProfile>("hfrun", title, nbin, runMin_ - 0.5, runMax_ + 0.5, 0.0, 1.0);
  for (int idet = 1; idet <= 4; idet++) {
    sprintf(name, "%s", hc[idet].c_str());
    sprintf(title, "Noise distribution for %s", hc[idet].c_str());
    h_[idet - 1] = fs_->make<TH1D>(name, title, 48, -6., 6.);
  }

  for (const auto& hcalid : hcalID_) {
    HcalDetId id = HcalDetId(hcalid);
    int subdet = id.subdetId();
    sprintf(name, "%s%d_%d_%d", hc[subdet].c_str(), id.ieta(), id.iphi(), id.depth());
    sprintf(title,
            "Energy Distribution for %s ieta %d iphi %d depth %d",
            hc[subdet].c_str(),
            id.ieta(),
            id.iphi(),
            id.depth());
    double xmin = (subdet == 4) ? -10 : -1;
    double xmax = (subdet == 4) ? 90 : 9;
    TH1D* hh = fs_->make<TH1D>(name, title, 50, xmin, xmax);
    histo_.push_back(hh);
  };

  if (extraHist_) {
    h_AmplitudeHBtest_ = fs_->make<TH1D>("h_AmplitudeHBtest", "", 5000, 0., 5000.);
    h_AmplitudeHEtest_ = fs_->make<TH1D>("h_AmplitudeHEtest", "", 3000, 0., 3000.);
    h_AmplitudeHFtest_ = fs_->make<TH1D>("h_AmplitudeHFtest", "", 10000, 0., 10000.);
    h_AmplitudeHB_ = fs_->make<TH1D>("h_AmplitudeHB", "", 100000, 0., 100000.);
    h_AmplitudeHE_ = fs_->make<TH1D>("h_AmplitudeHE", "", 300000, 0., 300000.);
    h_AmplitudeHF_ = fs_->make<TH1D>("h_AmplitudeHF", "", 100000, 0., 1000000.);
  }

  if (!fillHist_) {
    myTree_ = fs_->make<TTree>("RecJet", "RecJet Tree");
    myTree_->Branch("cells", &cells, "cells/I");
    myTree_->Branch("mysubd", &mysubd, "mysubd/I");
    myTree_->Branch("depth", &depth, "depth/I");
    myTree_->Branch("ieta", &ieta, "ieta/I");
    myTree_->Branch("iphi", &iphi, "iphi/I");
    myTree_->Branch("mom0_MB", &mom0_MB, "mom0_MB/F");
    myTree_->Branch("mom1_MB", &mom1_MB, "mom1_MB/F");
    myTree_->Branch("mom2_MB", &mom2_MB, "mom2_MB/F");
    myTree_->Branch("mom3_MB", &mom3_MB, "mom3_MB/F");
    myTree_->Branch("mom4_MB", &mom4_MB, "mom4_MB/F");
    myTree_->Branch("trigbit", &trigbit, "trigbit/I");
    myTree_->Branch("rnnumber", &rnnumber, "rnnumber/D");
  }
  myTree1_ = fs_->make<TTree>("RecJet1", "RecJet1 Tree");
  myTree1_->Branch("rnnum_", &rnnum_, "rnnum_/D");
  myTree1_->Branch("HBHEsize", &HBHEsize, "HBHEsize/I");
  myTree1_->Branch("HFsize", &HFsize, "HFsize/I");

  myMap_.clear();
}

//  EndJob
//
void RecAnalyzerMinbias::endJob() {
  if (!fillHist_) {
    cells = 0;
    for (const auto& itr : myMap_) {
      edm::LogVerbatim("RecAnalyzer") << "Fired trigger bit number " << itr.first.first;
      myInfo info = itr.second;
      if (info.theMB0 > 0) {
        mom0_MB = info.theMB0;
        mom1_MB = info.theMB1;
        mom2_MB = info.theMB2;
        mom3_MB = info.theMB3;
        mom4_MB = info.theMB4;
        rnnumber = info.runcheck;
        trigbit = itr.first.first;
        mysubd = itr.first.second.subdet();
        depth = itr.first.second.depth();
        iphi = itr.first.second.iphi();
        ieta = itr.first.second.ieta();
        edm::LogVerbatim("RecAnalyzer") << " Result=  " << trigbit << " " << mysubd << " " << ieta << " " << iphi
                                        << " mom0  " << mom0_MB << " mom1 " << mom1_MB << " mom2 " << mom2_MB
                                        << " mom3 " << mom3_MB << " mom4 " << mom4_MB;
        myTree_->Fill();
        cells++;
      }
    }
    edm::LogVerbatim("RecAnalyzer") << "cells"
                                    << " " << cells;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("RecAnalyzer") << "Exiting from RecAnalyzerMinbias::endjob";
#endif
}

void RecAnalyzerMinbias::beginRun(edm::Run const&, edm::EventSetup const& iS) {
  if (!init_) {
    init_ = true;
    if (fillHist_) {
      const HcalTopology* hcaltopology = &iS.getData(tok_htopo_);

      char name[700], title[700];
      // For HB
      int maxDepthHB = hcaltopology->maxDepthHB();
      int nbinHB = (Noise_) ? 18 : int(2000 * eHighHB_);
      double x_min = (Noise_) ? -3. : 0.;
      double x_max = (Noise_) ? 3. : 2. * eHighHB_;
      for (int eta = -50; eta < 50; eta++) {
        for (int phi = 0; phi < 100; phi++) {
          for (int depth = 1; depth <= maxDepthHB; depth++) {
            HcalDetId cell(HcalBarrel, eta, phi, depth);
            if (hcaltopology->valid(cell)) {
              sprintf(name, "HBeta%dphi%ddep%d", eta, phi, depth);
              sprintf(title, "HB #eta %d #phi %d depth %d", eta, phi, depth);
              TH1D* h = fs_->make<TH1D>(name, title, nbinHB, x_min, x_max);
              histHC_[cell] = h;
            }
          }
        }
      }
      // For HE
      int maxDepthHE = hcaltopology->maxDepthHE();
      int nbinHE = (Noise_) ? 18 : int(2000 * eHighHE_);
      x_min = (Noise_) ? -3. : 0.;
      x_max = (Noise_) ? 3. : 2. * eHighHE_;
      for (int eta = -50; eta < 50; eta++) {
        for (int phi = 0; phi < 100; phi++) {
          for (int depth = 1; depth <= maxDepthHE; depth++) {
            HcalDetId cell(HcalEndcap, eta, phi, depth);
            if (hcaltopology->valid(cell)) {
              sprintf(name, "HEeta%dphi%ddep%d", eta, phi, depth);
              sprintf(title, "HE #eta %d #phi %d depth %d", eta, phi, depth);
              TH1D* h = fs_->make<TH1D>(name, title, nbinHE, x_min, x_max);
              histHC_[cell] = h;
            }
          }
        }
      }
      // For HF
      int maxDepthHF = 4;
      int nbinHF = (Noise_) ? 200 : int(2000 * eHighHF_);
      x_min = (Noise_) ? -10. : 0.;
      x_max = (Noise_) ? 10. : 2. * eHighHF_;
      for (int eta = -50; eta < 50; eta++) {
        for (int phi = 0; phi < 100; phi++) {
          for (int depth = 1; depth <= maxDepthHF; depth++) {
            HcalDetId cell(HcalForward, eta, phi, depth);
            if (hcaltopology->valid(cell)) {
              sprintf(name, "HFeta%dphi%ddep%d", eta, phi, depth);
              sprintf(title, "Energy (HF #eta %d #phi %d depth %d)", eta, phi, depth);
              TH1D* h = fs_->make<TH1D>(name, title, nbinHF, x_min, x_max);
              histHC_[cell] = h;
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

  if (extraHist_) {
    double amplitudefullHB(0), amplitudefullHE(0), amplitudefullHF(0);
    edm::Handle<HBHEDigiCollection> hbhedigi;
    iEvent.getByToken(tok_hbhedigi_, hbhedigi);
    if (hbhedigi.isValid()) {
      for (auto const& digi : *(hbhedigi.product())) {
        int nTS = digi.size();
        double amplitudefullTSs = 0.;
        if (digi.id().subdet() == HcalBarrel) {
          if (nTS <= 10) {
            for (int i = 0; i < nTS; i++)
              amplitudefullTSs += digi.sample(i).adc();
            h_AmplitudeHBtest_->Fill(amplitudefullTSs);
            amplitudefullHB += amplitudefullTSs;
          }
        }
        if (digi.id().subdet() == HcalEndcap) {
          if (nTS <= 10) {
            for (int i = 0; i < nTS; i++)
              amplitudefullTSs += digi.sample(i).adc();
            h_AmplitudeHEtest_->Fill(amplitudefullTSs);
            amplitudefullHE += amplitudefullTSs;
          }
        }
      }
    }

    edm::Handle<QIE11DigiCollection> qie11digi;
    iEvent.getByToken(tok_qie11digi_, qie11digi);
    if (qie11digi.isValid()) {
      for (QIE11DataFrame const digi : *(qie11digi.product())) {
        double amplitudefullTSs = 0.;
        if (HcalDetId(digi.id()).subdet() == HcalBarrel) {
          for (int i = 0; i < digi.samples(); i++)
            amplitudefullTSs += digi[i].adc();
          h_AmplitudeHBtest_->Fill(amplitudefullTSs);
          amplitudefullHB += amplitudefullTSs;
        }
        if (HcalDetId(digi.id()).subdet() == HcalEndcap) {
          for (int i = 0; i < digi.samples(); i++)
            amplitudefullTSs += digi[i].adc();
          h_AmplitudeHEtest_->Fill(amplitudefullTSs);
          amplitudefullHE += amplitudefullTSs;
        }
      }
    }

    edm::Handle<HFDigiCollection> hfdigi;
    iEvent.getByToken(tok_hfdigi_, hfdigi);
    if (hfdigi.isValid()) {
      for (auto const& digi : *(hfdigi.product())) {
        int nTS = digi.size();
        double amplitudefullTSs = 0.;
        if (digi.id().subdet() == HcalForward) {
          if (nTS <= 10) {
            for (int i = 0; i < nTS; i++)
              amplitudefullTSs += digi.sample(i).adc();
            h_AmplitudeHFtest_->Fill(amplitudefullTSs);
            amplitudefullHF += amplitudefullTSs;
          }
        }
      }
    }

    edm::Handle<QIE10DigiCollection> qie10digi;
    iEvent.getByToken(tok_qie10digi_, qie10digi);
    if (qie10digi.isValid()) {
      for (QIE10DataFrame const digi : *(qie10digi.product())) {
        double amplitudefullTSs = 0.;
        if (HcalDetId(digi.id()).subdet() == HcalForward) {
          for (int i = 0; i < digi.samples(); i++)
            amplitudefullTSs += digi[i].adc();
          h_AmplitudeHFtest_->Fill(amplitudefullTSs);
          amplitudefullHF += amplitudefullTSs;
        }
      }
    }

    h_AmplitudeHB_->Fill(amplitudefullHB);
    h_AmplitudeHE_->Fill(amplitudefullHE);
    h_AmplitudeHF_->Fill(amplitudefullHF);
  }

  edm::Handle<HBHERecHitCollection> hbheMB;
  iEvent.getByToken(tok_hbherecoMB_, hbheMB);
  if (!hbheMB.isValid()) {
    edm::LogWarning("RecAnalyzer") << "HcalCalibAlgos: Error! can't get hbhe product!";
    return;
  }
  const HBHERecHitCollection HithbheMB = *(hbheMB.product());
  HBHEsize = HithbheMB.size();
  edm::LogVerbatim("RecAnalyzer") << "HBHE MB size of collection " << HithbheMB.size();
  if (HithbheMB.size() < 5100 && runNZS_) {
    edm::LogWarning("RecAnalyzer") << "HBHE problem " << rnnum_ << " size " << HBHEsize;
  }

  edm::Handle<HFRecHitCollection> hfMB;
  iEvent.getByToken(tok_hfrecoMB_, hfMB);
  if (!hfMB.isValid()) {
    edm::LogWarning("RecAnalyzer") << "HcalCalibAlgos: Error! can't get hf product!";
    return;
  }
  const HFRecHitCollection HithfMB = *(hfMB.product());
  edm::LogVerbatim("RecAnalyzer") << "HF MB size of collection " << HithfMB.size();
  HFsize = HithfMB.size();
  if (HithfMB.size() < 1700 && runNZS_) {
    edm::LogWarning("RecAnalyzer") << "HF problem " << rnnum_ << " size " << HFsize;
  }

  bool select(false);
  if (!trigbit_.empty()) {
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByToken(tok_hltL1GtMap_, gtObjectMapRecord);
    if (gtObjectMapRecord.isValid()) {
      const std::vector<L1GlobalTriggerObjectMap>& objMapVec = gtObjectMapRecord->gtObjectMap();
      for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin(); itMap != objMapVec.end();
           ++itMap) {
        bool resultGt = (*itMap).algoGtlResult();
        if (resultGt) {
          int algoBit = (*itMap).algoBitNumber();
          if (std::find(trigbit_.begin(), trigbit_.end(), algoBit) != trigbit_.end()) {
            select = true;
            break;
          }
        }
      }
    }
  }

  if (!trigbit_.empty() || select)
    myTree1_->Fill();

  //event weight for FLAT sample and PU information
  double eventWeight = 1.0;
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(tok_ew_, genEventInfo);
  if (genEventInfo.isValid())
    eventWeight = genEventInfo->weight();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("RecAnalyzer") << "Test HB " << HBHEsize << " HF " << HFsize << " Trigger " << trigbit_.size() << ":"
                                  << select << ":" << ignoreL1_ << " Wt " << eventWeight;
#endif
  if (ignoreL1_ || (!trigbit_.empty() && select)) {
    analyzeHcal(HithbheMB, HithfMB, 1, true, eventWeight);
  } else if ((!ignoreL1_) && (trigbit_.empty())) {
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByToken(tok_hltL1GtMap_, gtObjectMapRecord);
    if (gtObjectMapRecord.isValid()) {
      const std::vector<L1GlobalTriggerObjectMap>& objMapVec = gtObjectMapRecord->gtObjectMap();
      bool ok(false);
      for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin(); itMap != objMapVec.end();
           ++itMap) {
        bool resultGt = (*itMap).algoGtlResult();
        if (resultGt) {
          int algoBit = (*itMap).algoBitNumber();
          analyzeHcal(HithbheMB, HithfMB, algoBit, (!ok), eventWeight);
          ok = true;
        }
      }
      if (!ok) {
        edm::LogVerbatim("RecAnalyzer") << "No passed L1 Trigger found";
      }
    }
  }
}

void RecAnalyzerMinbias::analyzeHcal(
    const HBHERecHitCollection& HithbheMB, const HFRecHitCollection& HithfMB, int algoBit, bool fill, double weight) {
  // Signal part for HB HE
  int count(0), countHB(0), countHE(0), count2(0), count2HB(0), count2HE(0);
  for (HBHERecHitCollection::const_iterator hbheItr = HithbheMB.begin(); hbheItr != HithbheMB.end(); hbheItr++) {
    // Recalibration of energy
    DetId mydetid = hbheItr->id().rawId();
    double icalconst(1.);
    if (theRecalib_) {
      std::map<DetId, double>::iterator itr = corrFactor_.find(mydetid);
      if (itr != corrFactor_.end())
        icalconst = itr->second;
    }
    HBHERecHit aHit(hbheItr->id(), hbheItr->energy() * icalconst, hbheItr->time());
    double energyhit = aHit.energy();
    DetId id = (*hbheItr).detid();
    HcalDetId hid = HcalDetId(id);
    double eLow = (hid.subdet() == HcalEndcap) ? eLowHE_ : eLowHB_;
    double eHigh = (hid.subdet() == HcalEndcap) ? eHighHE_ : eHighHB_;
    ++count;
    if (id.subdetId() == HcalBarrel)
      ++countHB;
    else
      ++countHE;
    if (fill) {
      for (unsigned int i = 0; i < hcalID_.size(); i++) {
        if (hcalID_[i] == id.rawId()) {
          histo_[i]->Fill(energyhit);
          break;
        }
      }
      if (fillHist_) {
        std::map<HcalDetId, TH1D*>::iterator itr1 = histHC_.find(hid);
        if (itr1 != histHC_.end())
          itr1->second->Fill(energyhit);
      }
      h_[hid.subdet() - 1]->Fill(energyhit);
      if (energyhit > eMin_) {
        hbhe_->Fill(hid.ieta(), hid.iphi());
        ++count2;
        if (id.subdetId() == HcalBarrel) {
          ++count2HB;
          hb_->Fill(hid.ieta(), hid.iphi());
        } else {
          ++count2HE;
          he_->Fill(hid.ieta(), hid.iphi());
        }
      }
    }
    if (!fillHist_) {
      if (Noise_ || runNZS_ || (energyhit >= eLow && energyhit <= eHigh)) {
        std::map<std::pair<int, HcalDetId>, myInfo>::iterator itr1 =
            myMap_.find(std::pair<int, HcalDetId>(algoBit, hid));
        if (itr1 == myMap_.end()) {
          myInfo info;
          myMap_[std::pair<int, HcalDetId>(algoBit, hid)] = info;
          itr1 = myMap_.find(std::pair<int, HcalDetId>(algoBit, hid));
        }
        itr1->second.theMB0 += weight;
        itr1->second.theMB1 += (weight * energyhit);
        itr1->second.theMB2 += (weight * energyhit * energyhit);
        itr1->second.theMB3 += (weight * energyhit * energyhit * energyhit);
        itr1->second.theMB4 += (weight * energyhit * energyhit * energyhit * energyhit);
        itr1->second.runcheck = rnnum_;
      }
    }
  }  // HBHE_MB
  if (fill) {
    if (count > 0)
      hbherun_->Fill(rnnum_, (double)(count2) / count);
    if (countHB > 0)
      hbrun_->Fill(rnnum_, (double)(count2HB) / countHB);
    if (countHE > 0)
      herun_->Fill(rnnum_, (double)(count2HE) / countHE);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("RecAnalyzer") << "HBHE " << count2 << ":" << count << ":" << (double)(count2) / count << "\t HB "
                                  << count2HB << ":" << countHB << ":" << (double)(count2HB) / countHB << "\t HE "
                                  << count2HE << ":" << countHE << ":" << (double)(count2HE) / countHE;
#endif
  int countHF(0), count2HF(0);
  // Signal part for HF
  for (HFRecHitCollection::const_iterator hfItr = HithfMB.begin(); hfItr != HithfMB.end(); hfItr++) {
    // Recalibration of energy
    DetId mydetid = hfItr->id().rawId();
    double icalconst(1.);
    if (theRecalib_) {
      std::map<DetId, double>::iterator itr = corrFactor_.find(mydetid);
      if (itr != corrFactor_.end())
        icalconst = itr->second;
    }
    HFRecHit aHit(hfItr->id(), hfItr->energy() * icalconst, hfItr->time());

    double energyhit = aHit.energy();
    DetId id = (*hfItr).detid();
    HcalDetId hid = HcalDetId(id);
    ++countHF;
    if (fill) {
      for (unsigned int i = 0; i < hcalID_.size(); i++) {
        if (hcalID_[i] == id.rawId()) {
          histo_[i]->Fill(energyhit);
          break;
        }
      }
      if (fillHist_) {
        std::map<HcalDetId, TH1D*>::iterator itr1 = histHC_.find(hid);
        if (itr1 != histHC_.end())
          itr1->second->Fill(energyhit);
      }
      h_[hid.subdet() - 1]->Fill(energyhit);
      if (energyhit > eMin_) {
        hf_->Fill(hid.ieta(), hid.iphi());
        ++count2HF;
      }
    }

    //
    // Remove PMT hits
    //
    if (!fillHist_) {
      if (((Noise_ || runNZS_) && fabs(energyhit) <= 40.) || (energyhit >= eLowHF_ && energyhit <= eHighHF_)) {
        std::map<std::pair<int, HcalDetId>, myInfo>::iterator itr1 =
            myMap_.find(std::pair<int, HcalDetId>(algoBit, hid));
        if (itr1 == myMap_.end()) {
          myInfo info;
          myMap_[std::pair<int, HcalDetId>(algoBit, hid)] = info;
          itr1 = myMap_.find(std::pair<int, HcalDetId>(algoBit, hid));
        }
        itr1->second.theMB0 += weight;
        itr1->second.theMB1 += (weight * energyhit);
        itr1->second.theMB2 += (weight * energyhit * energyhit);
        itr1->second.theMB3 += (weight * energyhit * energyhit * energyhit);
        itr1->second.theMB4 += (weight * energyhit * energyhit * energyhit * energyhit);
        itr1->second.runcheck = rnnum_;
      }
    }
  }
  if (fill && countHF > 0)
    hfrun_->Fill(rnnum_, (double)(count2HF) / countHF);
#ifdef EDM_ML_DEBUG
  if (count)
    edm::LogVerbatim("RecAnalyzer") << "HF " << count2HF << ":" << countHF << ":" << (double)(count2HF) / countHF;
#endif
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecAnalyzerMinbias);
