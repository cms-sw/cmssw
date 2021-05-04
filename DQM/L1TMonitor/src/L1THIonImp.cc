#include "DQM/L1TMonitor/interface/L1THIonImp.h"

using namespace edm;

const unsigned int JETETABINS = 22;
const float JETETAMIN = -0.5;
const float JETETAMAX = 21.5;

const unsigned int EMETABINS = 22;
const float EMETAMIN = -0.5;
const float EMETAMAX = 21.5;

const unsigned int METPHIBINS = 72;
const float METPHIMIN = -0.5;
const float METPHIMAX = 71.5;

const unsigned int PHIBINS = 18;
const float PHIMIN = -0.5;
const float PHIMAX = 17.5;

const unsigned int OFBINS = 2;
const float OFMIN = -0.5;
const float OFMAX = 1.5;

const unsigned int BXBINS = 5;
const float BXMIN = -2.5;
const float BXMAX = 2.5;

// Bins for 3, 5, 6, 7, 10 and 12 bits
const unsigned int R3BINS = 8;
const float R3MIN = -0.5;
const float R3MAX = 7.5;
const unsigned int R6BINS = 64;
const float R6MIN = -0.5;
const float R6MAX = 63.5;
const unsigned int R12BINS = 4096;
const float R12MIN = -0.5;
const float R12MAX = 4095.5;

L1THIonImp::L1THIonImp(const edm::ParameterSet& ps)
    :  // data
      gctCenJetsDataSource_(ps.getParameter<edm::InputTag>("gctCentralJetsDataSource")),
      gctForJetsDataSource_(ps.getParameter<edm::InputTag>("gctForwardJetsDataSource")),
      gctTauJetsDataSource_(ps.getParameter<edm::InputTag>("gctTauJetsDataSource")),
      gctEnergySumsDataSource_(ps.getParameter<edm::InputTag>("gctEnergySumsDataSource")),
      gctIsoEmDataSource_(ps.getParameter<edm::InputTag>("gctIsoEmDataSource")),
      gctNonIsoEmDataSource_(ps.getParameter<edm::InputTag>("gctNonIsoEmDataSource")),
      // RCT
      rctSource_L1CRCollection_(consumes<L1CaloRegionCollection>(ps.getParameter<InputTag>("rctSource"))),
      //  emul
      gctCenJetsEmulSource_(ps.getParameter<edm::InputTag>("gctCentralJetsEmulSource")),
      gctForJetsEmulSource_(ps.getParameter<edm::InputTag>("gctForwardJetsEmulSource")),
      gctTauJetsEmulSource_(ps.getParameter<edm::InputTag>("gctTauJetsEmulSource")),
      gctEnergySumsEmulSource_(ps.getParameter<edm::InputTag>("gctEnergySumsEmulSource")),
      gctIsoEmEmulSource_(ps.getParameter<edm::InputTag>("gctIsoEmEmulSource")),
      gctNonIsoEmEmulSource_(ps.getParameter<edm::InputTag>("gctNonIsoEmEmulSource")) {
  //set Token(-s)
  gctIsoEmSourceDataToken_ = consumes<L1GctEmCandCollection>(ps.getParameter<edm::InputTag>("gctIsoEmDataSource"));
  gctNonIsoEmSourceDataToken_ =
      consumes<L1GctEmCandCollection>(ps.getParameter<edm::InputTag>("gctNonIsoEmDataSource"));
  gctCenJetsSourceDataToken_ =
      consumes<L1GctJetCandCollection>(ps.getParameter<edm::InputTag>("gctCentralJetsDataSource"));
  gctForJetsSourceDataToken_ =
      consumes<L1GctJetCandCollection>(ps.getParameter<edm::InputTag>("gctForwardJetsDataSource"));
  gctTauJetsSourceDataToken_ = consumes<L1GctJetCandCollection>(ps.getParameter<edm::InputTag>("gctTauJetsDataSource"));
  gctEnergySumsSourceDataToken_ =
      consumes<L1GctHFRingEtSumsCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsDataSource"));
  l1HFCountsDataToken_ =
      consumes<L1GctHFBitCountsCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsDataSource"));
  l1EtMissDataToken_ = consumes<L1GctEtMissCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsDataSource"));
  l1HtMissDataToken_ = consumes<L1GctHtMissCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsDataSource"));
  l1EtHadDataToken_ = consumes<L1GctEtHadCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsDataSource"));
  l1EtTotalDataToken_ = consumes<L1GctEtTotalCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsDataSource"));

  gctIsoEmSourceEmulToken_ = consumes<L1GctEmCandCollection>(ps.getParameter<edm::InputTag>("gctIsoEmEmulSource"));
  gctNonIsoEmSourceEmulToken_ =
      consumes<L1GctEmCandCollection>(ps.getParameter<edm::InputTag>("gctNonIsoEmEmulSource"));
  gctCenJetsSourceEmulToken_ =
      consumes<L1GctJetCandCollection>(ps.getParameter<edm::InputTag>("gctCentralJetsEmulSource"));
  gctForJetsSourceEmulToken_ =
      consumes<L1GctJetCandCollection>(ps.getParameter<edm::InputTag>("gctForwardJetsEmulSource"));
  gctTauJetsSourceEmulToken_ = consumes<L1GctJetCandCollection>(ps.getParameter<edm::InputTag>("gctTauJetsEmulSource"));
  gctEnergySumsSourceEmulToken_ =
      consumes<L1GctHFRingEtSumsCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsEmulSource"));
  l1HFCountsEmulToken_ =
      consumes<L1GctHFBitCountsCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsEmulSource"));
  l1EtMissEmulToken_ = consumes<L1GctEtMissCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsEmulSource"));
  l1HtMissEmulToken_ = consumes<L1GctHtMissCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsEmulSource"));
  l1EtHadEmulToken_ = consumes<L1GctEtHadCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsEmulSource"));
  l1EtTotalEmulToken_ = consumes<L1GctEtTotalCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsEmulSource"));
}

L1THIonImp::~L1THIonImp() {}

void L1THIonImp::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder("L1T/L1THIon");

  l1GctCenJetsEtEtaPhi_ =
      ibooker.book2D("CenJetsEtEtaPhi", "CENTRAL JET E_{T}", JETETABINS, JETETAMIN, JETETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctForJetsEtEtaPhi_ =
      ibooker.book2D("ForJetsEtEtaPhi", "FORWARD JET E_{T}", JETETABINS, JETETAMIN, JETETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctTauJetsEtEtaPhi_ = ibooker.book2D(
      "SingleTrackTriggerEtEtaPhi", "TAU JET E_{T}", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctIsoEmRankEtaPhi_ =
      ibooker.book2D("IsoEmRankEtaPhi", "ISO EM E_{T}", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctNonIsoEmRankEtaPhi_ =
      ibooker.book2D("NonIsoEmRankEtaPhi", "NON-ISO EM E_{T}", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX);

  l1GctCenJetsOccEtaPhi_ = ibooker.book2D(
      "CenJetsOccEtaPhi", "CENTRAL JET OCCUPANCY", JETETABINS, JETETAMIN, JETETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctForJetsOccEtaPhi_ = ibooker.book2D(
      "ForJetsOccEtaPhi", "FORWARD JET OCCUPANCY", JETETABINS, JETETAMIN, JETETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctTauJetsOccEtaPhi_ = ibooker.book2D(
      "SingleTrackTriggerOccEtaPhi", "TAU JET OCCUPANCY", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctIsoEmOccEtaPhi_ =
      ibooker.book2D("IsoEmOccEtaPhi", "ISO EM OCCUPANCY", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctNonIsoEmOccEtaPhi_ = ibooker.book2D(
      "NonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX);

  l1GctHFRing1TowerCountPosEtaNegEta_ = ibooker.book2D(
      "HFRing1TowerCountCorr", "HF RING1 TOWER COUNT CORRELATION +/-  #eta", R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX);
  l1GctHFRing2TowerCountPosEtaNegEta_ = ibooker.book2D(
      "HFRing2TowerCountCorr", "HF RING2 TOWER COUNT CORRELATION +/-  #eta", R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX);

  l1GctHFRing1TowerCountPosEta_ =
      ibooker.book1D("HFRing1TowerCountPosEta", "HF RING1 TOWER COUNT  #eta  +", R3BINS, R3MIN, R3MAX);
  l1GctHFRing1TowerCountNegEta_ =
      ibooker.book1D("HFRing1TowerCountNegEta", "HF RING1 TOWER COUNT  #eta  -", R3BINS, R3MIN, R3MAX);
  l1GctHFRing2TowerCountPosEta_ =
      ibooker.book1D("HFRing2TowerCountPosEta", "HF RING2 TOWER COUNT  #eta  +", R3BINS, R3MIN, R3MAX);
  l1GctHFRing2TowerCountNegEta_ =
      ibooker.book1D("HFRing2TowerCountNegEta", "HF RING2 TOWER COUNT  #eta  -", R3BINS, R3MIN, R3MAX);

  l1GctHFRingTowerCountOccBx_ =
      ibooker.book2D("HFRingTowerCountOccBx", "HF RING TOWER COUNT PER BX", BXBINS, BXMIN, BXMAX, R3BINS, R3MIN, R3MAX);

  l1GctHFRing1PosEtaNegEta_ = ibooker.book2D("centrality and centrality ext Corr",
                                             "centrality and centrality ext E_{T} CORRELATION",
                                             R3BINS,
                                             R3MIN,
                                             R3MAX,
                                             R3BINS,
                                             R3MIN,
                                             R3MAX);
  l1GctHFRing1ETSumPosEta_ = ibooker.book1D("centrality", "centrality E_{T}", 8, -0.5, 7.5);
  l1GctHFRing1ETSumNegEta_ = ibooker.book1D("centrality ext", "centrality ext E_{T}", 8, -0.5, 7.5);
  l1GctHFRingETSum_ = ibooker.book1D("centrality+centralityExt Et", "centrality+centralityExt E_{T}", 8, -0.5, 7.5);
  l1GctHFRingETDiff_ = ibooker.book1D("centrality-centralityExt Et", "centrality-centralityExt E_{T}", 8, -0.5, 7.5);

  l1GctHFRingETSumOccBx_ =
      ibooker.book2D("centrality OccBx", "centrality PER BX", BXBINS, BXMIN, BXMAX, R3BINS, R3MIN, R3MAX);
  l1GctHFRingRatioPosEta_ =
      ibooker.book1D("centrality centralityExt ratio", "centrality centralityExt ratio", 9, -0.5, 8.5);

  l1GctMinBiasBitHFEt_ = ibooker.book1D("HI Minimum Bias bits HF Et", "HI Minimum Bias bits HF Et", 6, -0.5, 5.5);

  l1GctCenJetsRank_ = ibooker.book1D("CenJetsRank", "CENTRAL JET E_{T}", R6BINS, R6MIN, R6MAX);
  l1GctForJetsRank_ = ibooker.book1D("ForJetsRank", "FORWARD JET E_{T}", R6BINS, R6MIN, R6MAX);
  l1GctTauJetsRank_ = ibooker.book1D("SingleTrackTriggerRank", "Single Track Trigger E_{T}", R6BINS, R6MIN, R6MAX);
  l1GctIsoEmRank_ = ibooker.book1D("IsoEmRank", "ISO EM E_{T}", R6BINS, R6MIN, R6MAX);
  l1GctNonIsoEmRank_ = ibooker.book1D("NonIsoEmRank", "NON-ISO EM E_{T}", R6BINS, R6MIN, R6MAX);

  l1GctAllJetsOccRankBx_ =
      ibooker.book2D("AllJetsOccRankBx", "ALL JETS E_{T} PER BX", BXBINS, BXMIN, BXMAX, R6BINS, R6MIN, R6MAX);
  l1GctAllEmOccRankBx_ =
      ibooker.book2D("AllEmOccRankBx", "ALL EM E_{T} PER BX", BXBINS, BXMIN, BXMAX, R6BINS, R6MIN, R6MAX);

  l1GctEtMiss_ = ibooker.book1D("EtMiss", "MET", R12BINS, R12MIN, R12MAX);
  l1GctEtMissPhi_ = ibooker.book1D("EtMissPhi", "MET  #phi", METPHIBINS, METPHIMIN, METPHIMAX);
  l1GctEtMissOf_ = ibooker.book1D("EtMissOf", "MET OVERFLOW", OFBINS, OFMIN, OFMAX);
  l1GctEtMissOccBx_ = ibooker.book2D("EtMissOccBx", "MET PER BX", BXBINS, BXMIN, BXMAX, R12BINS, R12MIN, R12MAX);

  l1GctEtTotal_ = ibooker.book1D("EtTotal", "SUM E_{T}", R12BINS, R12MIN, R12MAX);
  l1GctEtTotalOf_ = ibooker.book1D("EtTotalOf", "SUM E_{T} OVERFLOW", OFBINS, OFMIN, OFMAX);
  l1GctEtTotalOccBx_ =
      ibooker.book2D("EtTotalOccBx", "SUM E_{T} PER BX", BXBINS, BXMIN, BXMAX, R12BINS, R12MIN, R12MAX);

  l1GctEtHad_ = ibooker.book1D("EtHad", "H_{T}", R12BINS, R12MIN, R12MAX);
  l1GctEtHadOf_ = ibooker.book1D("EtHadOf", "H_{T} OVERFLOW", OFBINS, OFMIN, OFMAX);
  l1GctEtHadOccBx_ = ibooker.book2D("EtHadOccBx", "H_{T} PER BX", BXBINS, BXMIN, BXMAX, R12BINS, R12MIN, R12MAX);

  l1GctEtTotalEtHadCorr_ =
      ibooker.book2D("EtTotalEtHadCorr", "Sum E_{T} H_{T} CORRELATION", R6BINS, R12MIN, R12MAX, R6BINS, R12MIN, R12MAX);

  HFPosEnergy_ = ibooker.book1D("HF+ Energy Sum", "HF+ Energy Sum", R12BINS, R12MIN, R12MAX);
  HFNegEnergy_ = ibooker.book1D("HF- Energy Sum", "HF- Energy Sum", R12BINS, R12MIN, R12MAX);
  HFEnergy_ = ibooker.book1D("HF Energy Sum", "HF Energy Sum", R12BINS, R12MIN, R12MAX);

  ibooker.setCurrentFolder("L1TEMU/L1TEMUHIon");

  const std::string clabel[8] = {"cenJet", "forJet", "single track", "isoEm", "nonIsoEm", "EtSum", "MET", "HTT"};
  const std::string olabel[3] = {"rank", "eta", "phi"};

  unsigned int Bin[3][8] = {{64, 64, 64, 64, 64, 128, 128, 128},
                            {EMETABINS, EMETABINS, EMETABINS, EMETABINS, EMETABINS, EMETABINS, EMETABINS, EMETABINS},
                            {PHIBINS, PHIBINS, PHIBINS, PHIBINS, PHIBINS, PHIBINS, METPHIBINS, PHIBINS}};
  float Min[3][8] = {{-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5},
                     {-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5},
                     {-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5}};
  float Max[3][8] = {{63.5, 63.5, 63.5, 63.5, 63.5, 1023.5, 1023.5, 1023.5},
                     {21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5},
                     {17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 71.5, 17.5}};

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 8; j++) {
      DECorr_[i][j] = ibooker.book2D(clabel[j] + olabel[i] + " data vs emul",
                                     clabel[j] + olabel[i] + " data vs emul",
                                     Bin[i][j],
                                     Min[i][j],
                                     Max[i][j],
                                     Bin[i][j],
                                     Min[i][j],
                                     Max[i][j]);
    }
  }

  centralityCorr_ = ibooker.book2D("centrality data vs emul", "centrality data vs emul", 8, -0.5, 7.5, 8, -0.5, 7.5);
  centralityExtCorr_ =
      ibooker.book2D("centrality ext data vs emul", "centrality ext data vs emul", 8, -0.5, 7.5, 8, -0.5, 7.5);
  MinBiasCorr_ = ibooker.book2D(
      "Minimum Bias Trigger Data vs Emul", "Minimum Bias Trigger Data vs Emul", 6, -0.5, 5.5, 6, -0.5, 5.5);
}

void L1THIonImp::analyze(const edm::Event& e, const edm::EventSetup& c) {
  edm::Handle<L1GctEmCandCollection> l1IsoEm;
  edm::Handle<L1GctEmCandCollection> l1NonIsoEm;
  edm::Handle<L1GctJetCandCollection> l1CenJets;
  edm::Handle<L1GctJetCandCollection> l1ForJets;
  edm::Handle<L1GctJetCandCollection> l1TauJets;
  edm::Handle<L1GctHFRingEtSumsCollection> l1HFSums;
  edm::Handle<L1GctHFBitCountsCollection> l1HFCounts;
  edm::Handle<L1GctEtMissCollection> l1EtMiss;
  edm::Handle<L1GctHtMissCollection> l1HtMiss;
  edm::Handle<L1GctEtHadCollection> l1EtHad;
  edm::Handle<L1GctEtTotalCollection> l1EtTotal;

  edm::Handle<L1GctEmCandCollection> l1IsoEmEmul;
  edm::Handle<L1GctEmCandCollection> l1NonIsoEmEmul;
  edm::Handle<L1GctJetCandCollection> l1CenJetsEmul;
  edm::Handle<L1GctJetCandCollection> l1ForJetsEmul;
  edm::Handle<L1GctJetCandCollection> l1TauJetsEmul;
  edm::Handle<L1GctHFRingEtSumsCollection> l1HFSumsEmul;
  edm::Handle<L1GctHFBitCountsCollection> l1HFCountsEmul;
  edm::Handle<L1GctEtMissCollection> l1EtMissEmul;
  edm::Handle<L1GctHtMissCollection> l1HtMissEmul;
  edm::Handle<L1GctEtHadCollection> l1EtHadEmul;
  edm::Handle<L1GctEtTotalCollection> l1EtTotalEmul;

  edm::Handle<L1CaloRegionCollection> rgn;
  e.getByToken(rctSource_L1CRCollection_, rgn);

  e.getByToken(gctIsoEmSourceDataToken_, l1IsoEm);
  e.getByToken(gctNonIsoEmSourceDataToken_, l1NonIsoEm);
  e.getByToken(gctCenJetsSourceDataToken_, l1CenJets);
  e.getByToken(gctForJetsSourceDataToken_, l1ForJets);
  e.getByToken(gctTauJetsSourceDataToken_, l1TauJets);
  e.getByToken(gctEnergySumsSourceDataToken_, l1HFSums);
  e.getByToken(l1HFCountsDataToken_, l1HFCounts);
  e.getByToken(l1EtMissDataToken_, l1EtMiss);
  e.getByToken(l1HtMissDataToken_, l1HtMiss);
  e.getByToken(l1EtHadDataToken_, l1EtHad);
  e.getByToken(l1EtTotalDataToken_, l1EtTotal);

  e.getByToken(gctIsoEmSourceEmulToken_, l1IsoEmEmul);
  e.getByToken(gctNonIsoEmSourceEmulToken_, l1NonIsoEmEmul);
  e.getByToken(gctCenJetsSourceEmulToken_, l1CenJetsEmul);
  e.getByToken(gctForJetsSourceEmulToken_, l1ForJetsEmul);
  e.getByToken(gctTauJetsSourceEmulToken_, l1TauJetsEmul);
  e.getByToken(gctEnergySumsSourceEmulToken_, l1HFSumsEmul);
  e.getByToken(l1HFCountsEmulToken_, l1HFCountsEmul);
  e.getByToken(l1EtMissEmulToken_, l1EtMissEmul);
  e.getByToken(l1HtMissEmulToken_, l1HtMissEmul);
  e.getByToken(l1EtHadEmulToken_, l1EtHadEmul);
  e.getByToken(l1EtTotalEmulToken_, l1EtTotalEmul);

  // Fill histograms

  // Central jets

  for (L1GctJetCandCollection::const_iterator cj = l1CenJets->begin(); cj != l1CenJets->end(); cj++) {
    // only plot central BX
    if (cj->bx() == 0) {
      l1GctCenJetsRank_->Fill(cj->rank());
      // only plot eta and phi maps for non-zero candidates
      if (cj->rank()) {
        l1GctCenJetsEtEtaPhi_->Fill(cj->regionId().ieta(), cj->regionId().iphi(), cj->rank());
        l1GctCenJetsOccEtaPhi_->Fill(cj->regionId().ieta(), cj->regionId().iphi());
      }
    }
    if (cj->rank())
      l1GctAllJetsOccRankBx_->Fill(cj->bx(), cj->rank());  // for all BX
    for (L1GctJetCandCollection::const_iterator j = l1CenJetsEmul->begin(); j != l1CenJetsEmul->end(); j++) {
      if (cj->bx() == 0 && j->bx() == 0 &&
          std::distance(l1CenJets->begin(), cj) == std::distance(l1CenJetsEmul->begin(), j)) {
        //std::cout<<std::to_string(j)<<std::endl;
        DECorr_[0][0]->Fill(cj->rank(), j->rank());
        DECorr_[1][0]->Fill(cj->regionId().ieta(), j->regionId().ieta());
        DECorr_[2][0]->Fill(cj->regionId().iphi(), j->regionId().iphi());
      }
    }
  }
  for (L1GctJetCandCollection::const_iterator j = l1CenJetsEmul->begin(); j != l1CenJetsEmul->end(); j++) {
  }

  // Forward jets
  for (L1GctJetCandCollection::const_iterator fj = l1ForJets->begin(); fj != l1ForJets->end(); fj++) {
    // only plot central BX
    if (fj->bx() == 0) {
      l1GctForJetsRank_->Fill(fj->rank());
      // only plot eta and phi maps for non-zero candidates
      if (fj->rank()) {
        l1GctForJetsEtEtaPhi_->Fill(fj->regionId().ieta(), fj->regionId().iphi(), fj->rank());
        l1GctForJetsOccEtaPhi_->Fill(fj->regionId().ieta(), fj->regionId().iphi());
      }
    }
    if (fj->rank())
      l1GctAllJetsOccRankBx_->Fill(fj->bx(), fj->rank());  // for all BX
    for (L1GctJetCandCollection::const_iterator j = l1ForJetsEmul->begin(); j != l1ForJetsEmul->end(); j++) {
      if (fj->bx() == 0 && j->bx() == 0 &&
          std::distance(l1ForJets->begin(), fj) == std::distance(l1ForJetsEmul->begin(), j)) {
        DECorr_[0][1]->Fill(fj->rank(), j->rank());
        DECorr_[1][1]->Fill(fj->regionId().ieta(), j->regionId().ieta());
        DECorr_[2][1]->Fill(fj->regionId().iphi(), j->regionId().iphi());
      }
    }
  }

  for (L1GctJetCandCollection::const_iterator tj = l1TauJets->begin(); tj != l1TauJets->end(); tj++) {
    // only plot central BX
    if (tj->bx() == 0) {
      l1GctTauJetsRank_->Fill(tj->rank());
      // only plot eta and phi maps for non-zero candidates
      if (tj->rank()) {
        l1GctTauJetsEtEtaPhi_->Fill(tj->regionId().ieta(), tj->regionId().iphi(), tj->rank());
        l1GctTauJetsOccEtaPhi_->Fill(tj->regionId().ieta(), tj->regionId().iphi());
      }
    }
    if (tj->rank())
      l1GctAllJetsOccRankBx_->Fill(tj->bx(), tj->rank());  // for all BX
    for (L1GctJetCandCollection::const_iterator j = l1TauJetsEmul->begin(); j != l1TauJetsEmul->end(); j++) {
      if (tj->bx() == 0 && j->bx() == 0 &&
          std::distance(l1TauJets->begin(), tj) == std::distance(l1TauJetsEmul->begin(), j)) {
        DECorr_[0][2]->Fill(tj->rank(), j->rank());
        DECorr_[1][2]->Fill(tj->regionId().ieta(), j->regionId().ieta());
        DECorr_[2][2]->Fill(tj->regionId().iphi(), j->regionId().iphi());
      }
    }
  }

  for (L1GctEtMissCollection::const_iterator met = l1EtMiss->begin(); met != l1EtMiss->end(); met++) {
    // only plot central BX
    if (met->bx() == 0) {
      if (met->overFlow() == 0 && met->et() > 0) {
        //Avoid problems with met=0 candidates affecting MET_PHI plots
        l1GctEtMiss_->Fill(met->et());
        l1GctEtMissPhi_->Fill(met->phi());
      }
      l1GctEtMissOf_->Fill(met->overFlow());
    }
    if (met->overFlow() == 0 && met->et() > 0)
      l1GctEtMissOccBx_->Fill(met->bx(), met->et());  // for all BX
    for (L1GctEtMissCollection::const_iterator j = l1EtMissEmul->begin(); j != l1EtMissEmul->end(); j++) {
      if (met->bx() == 0 && j->bx() == 0) {
        DECorr_[0][6]->Fill(met->et(), j->et());
        DECorr_[2][6]->Fill(met->phi(), j->phi());
      }
    }
  }

  for (L1GctEtHadCollection::const_iterator ht = l1EtHad->begin(); ht != l1EtHad->end(); ht++) {
    // only plot central BX
    if (ht->bx() == 0) {
      l1GctEtHad_->Fill(ht->et());
      l1GctEtHadOf_->Fill(ht->overFlow());
    }
    l1GctEtHadOccBx_->Fill(ht->bx(), ht->et());  // for all BX
    for (L1GctEtHadCollection::const_iterator j = l1EtHadEmul->begin(); j != l1EtHadEmul->end(); j++) {
      if (ht->bx() == 0 && j->bx() == 0) {
        DECorr_[0][7]->Fill(ht->et(), j->et());
        //DECorr_[2][7]->Fill(ht->ieta(),j->ieta());
        //DECorr_[3][7]->Fill(ht->iphi(),j->iphi());
      }
    }
  }

  for (L1GctEtTotalCollection::const_iterator et = l1EtTotal->begin(); et != l1EtTotal->end(); et++) {
    // only plot central BX
    if (et->bx() == 0) {
      l1GctEtTotal_->Fill(et->et());
      l1GctEtTotalOf_->Fill(et->overFlow());
    }
    l1GctEtTotalOccBx_->Fill(et->bx(), et->et());  // for all BX
    for (L1GctEtTotalCollection::const_iterator j = l1EtTotalEmul->begin(); j != l1EtTotalEmul->end(); j++) {
      if (et->bx() == 0 && j->bx() == 0) {
        DECorr_[0][5]->Fill(et->et(), j->et());
        //DECorr_[2][5]->Fill(et->eta(),j->eta());
        //DECorr_[3][5]->Fill(et->iphi(),j->iphi());
      }
    }
  }

  for (L1GctEmCandCollection::const_iterator ie = l1IsoEm->begin(); ie != l1IsoEm->end(); ie++) {
    // only plot central BX
    if (ie->bx() == 0) {
      l1GctIsoEmRank_->Fill(ie->rank());
      // only plot eta and phi maps for non-zero candidates
      if (ie->rank()) {
        l1GctIsoEmRankEtaPhi_->Fill(ie->regionId().ieta(), ie->regionId().iphi(), ie->rank());
        l1GctIsoEmOccEtaPhi_->Fill(ie->regionId().ieta(), ie->regionId().iphi());
      }
    }
    if (ie->rank())
      l1GctAllEmOccRankBx_->Fill(ie->bx(), ie->rank());  // for all BX
    for (L1GctEmCandCollection::const_iterator j = l1IsoEmEmul->begin(); j != l1IsoEmEmul->end(); j++) {
      if (ie->bx() == 0 && j->bx() == 0 &&
          std::distance(l1IsoEm->begin(), ie) == std::distance(l1IsoEmEmul->begin(), j)) {
        DECorr_[0][3]->Fill(ie->rank(), j->rank());
        DECorr_[1][3]->Fill(ie->regionId().ieta(), j->regionId().ieta());
        DECorr_[2][3]->Fill(ie->regionId().iphi(), j->regionId().iphi());
      }
    }
  }

  for (L1GctEmCandCollection::const_iterator ne = l1NonIsoEm->begin(); ne != l1NonIsoEm->end(); ne++) {
    // only plot central BX
    if (ne->bx() == 0) {
      l1GctNonIsoEmRank_->Fill(ne->rank());
      // only plot eta and phi maps for non-zero candidates
      if (ne->rank()) {
        l1GctNonIsoEmRankEtaPhi_->Fill(ne->regionId().ieta(), ne->regionId().iphi(), ne->rank());
        l1GctNonIsoEmOccEtaPhi_->Fill(ne->regionId().ieta(), ne->regionId().iphi());
      }
    }
    if (ne->rank())
      l1GctAllEmOccRankBx_->Fill(ne->bx(), ne->rank());  // for all BX
    for (L1GctEmCandCollection::const_iterator j = l1NonIsoEmEmul->begin(); j != l1NonIsoEmEmul->end(); j++) {
      if (ne->bx() == 0 && j->bx() == 0 &&
          std::distance(l1NonIsoEm->begin(), ne) == std::distance(l1NonIsoEmEmul->begin(), j)) {
        DECorr_[0][4]->Fill(ne->rank(), j->rank());
        DECorr_[1][4]->Fill(ne->regionId().ieta(), j->regionId().ieta());
        DECorr_[2][4]->Fill(ne->regionId().iphi(), j->regionId().iphi());
      }
    }
  }

  for (L1GctHFBitCountsCollection::const_iterator hfc = l1HFCounts->begin(); hfc != l1HFCounts->end(); hfc++) {
    // only plot central BX
    if (hfc->bx() == 0) {
      // Individual ring counts
      l1GctHFRing1TowerCountPosEta_->Fill(hfc->bitCount(0));
      l1GctHFRing1TowerCountNegEta_->Fill(hfc->bitCount(1));
      l1GctHFRing2TowerCountPosEta_->Fill(hfc->bitCount(2));
      l1GctHFRing2TowerCountNegEta_->Fill(hfc->bitCount(3));
      // Correlate positive and negative eta
      l1GctHFRing1TowerCountPosEtaNegEta_->Fill(hfc->bitCount(0), hfc->bitCount(1));
      l1GctHFRing2TowerCountPosEtaNegEta_->Fill(hfc->bitCount(2), hfc->bitCount(3));
    }
    // Occupancy vs BX
    for (unsigned i = 0; i < 4; i++) {
      l1GctHFRingTowerCountOccBx_->Fill(hfc->bx(), hfc->bitCount(i));
    }
  }

  for (L1GctHFRingEtSumsCollection::const_iterator hfs = l1HFSums->begin(); hfs != l1HFSums->end(); hfs++) {
    if (hfs->bx() == 0) {
      l1GctHFRing1ETSumPosEta_->Fill(hfs->etSum(0));
      l1GctHFRing1ETSumNegEta_->Fill(hfs->etSum(1));
      l1GctHFRingETSum_->Fill(hfs->etSum(0) + hfs->etSum(1));
      l1GctHFRingETDiff_->Fill(abs(hfs->etSum(0) - hfs->etSum(1)));
      if (hfs->etSum(1) != 0)
        l1GctHFRingRatioPosEta_->Fill((hfs->etSum(0)) / (hfs->etSum(1)));
      l1GctHFRing1PosEtaNegEta_->Fill(hfs->etSum(0), hfs->etSum(1));
      std::vector<int> bit = SortMinBiasBit(hfs->etSum(2), hfs->etSum(3));
      for (std::vector<int>::const_iterator it = bit.begin(); it != bit.end(); it++) {
        l1GctMinBiasBitHFEt_->Fill(it - bit.begin(), *it);
      }
    }
    for (unsigned i = 0; i < 4; i++) {
      l1GctHFRingETSumOccBx_->Fill(hfs->bx(), hfs->etSum(i));
    }
    for (L1GctHFRingEtSumsCollection::const_iterator j = l1HFSumsEmul->begin(); j != l1HFSumsEmul->end(); j++) {
      if (hfs->bx() == 0 && j->bx() == 0 &&
          std::distance(l1HFSums->begin(), hfs) == std::distance(l1HFSumsEmul->begin(), j)) {
        centralityCorr_->Fill(hfs->etSum(0), j->etSum(0));
        centralityExtCorr_->Fill(hfs->etSum(1), j->etSum(1));
        std::vector<int> dbit = SortMinBiasBit(hfs->etSum(2), hfs->etSum(3));
        std::vector<int> ebit = SortMinBiasBit(j->etSum(2), j->etSum(3));
      }
    }
  }

  for (L1CaloRegionCollection::const_iterator it = rgn->begin(); it != rgn->end(); it++) {
    if (it->bx() == 0) {
      int totm = 0;
      int totp = 0;
      if (it->gctEta() < 4) {
        totm += it->et();
      }
      if (it->gctEta() > 17) {
        totp += it->et();
      }
      HFNegEnergy_->Fill(totm);
      HFPosEnergy_->Fill(totp);
      HFEnergy_->Fill(totm + totp);
    }
  }
}

std::vector<int> L1THIonImp::SortMinBiasBit(uint16_t a, uint16_t b) {
  std::vector<int> Bit;

  if ((a + 1) / 4 > 0.5) {
    Bit.push_back(1);
  } else {
    Bit.push_back(0);
  }

  if (a == 2 || a == 3 || a == 6 || a == 7) {
    Bit.push_back(1);
  } else {
    Bit.push_back(0);
  }

  Bit.push_back(a % 2);

  if ((b + 1) / 4 > 0.5) {
    Bit.push_back(1);
  } else {
    Bit.push_back(0);
  }

  if (b == 2 || b == 3 || b == 6 || b == 7) {
    Bit.push_back(1);
  } else {
    Bit.push_back(0);
  }

  Bit.push_back(b % 2);

  return Bit;
}
