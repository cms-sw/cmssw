#include "DQM/L1TMonitor/interface/L1TGCT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

// Trigger Headers

// GCT and RCT data formats
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace edm;

// Define statics for bins etc.
const unsigned int JETETABINS = 22;
const float JETETAMIN = -0.5;
const float JETETAMAX = 21.5;

const unsigned int EMETABINS = 22;
const float EMETAMIN = -0.5;
const float EMETAMAX = 21.5;

const unsigned int METPHIBINS = 72;
const float METPHIMIN = -0.5;
const float METPHIMAX = 71.5;

const unsigned int MHTPHIBINS = 18;
const float MHTPHIMIN = -0.5;
const float MHTPHIMAX = 17.5;

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
const unsigned int R5BINS = 32;
const float R5MIN = -0.5;
const float R5MAX = 31.5;
const unsigned int R6BINS = 64;
const float R6MIN = -0.5;
const float R6MAX = 63.5;
const unsigned int R7BINS = 128;
const float R7MIN = -0.5;
const float R7MAX = 127.5;
const unsigned int R12BINS = 4096;
const float R12MIN = -0.5;
const float R12MAX = 4095.5;

L1TGCT::L1TGCT(const edm::ParameterSet & ps) :
  monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir","")),
  gctCenJetsSource_(ps.getParameter<edm::InputTag>("gctCentralJetsSource")),
  gctForJetsSource_(ps.getParameter<edm::InputTag>("gctForwardJetsSource")),
  gctTauJetsSource_(ps.getParameter<edm::InputTag>("gctTauJetsSource")),
  gctIsoTauJetsSource_(ps.getParameter<edm::InputTag>("gctIsoTauJetsSource")),
  gctEnergySumsSource_(ps.getParameter<edm::InputTag>("gctEnergySumsSource")),
  gctIsoEmSource_(ps.getParameter<edm::InputTag>("gctIsoEmSource")),
  gctNonIsoEmSource_(ps.getParameter<edm::InputTag>("gctNonIsoEmSource")),
  m_stage1_layer2_(ps.getParameter<bool>("stage1_layer2_")),
  filterTriggerType_ (ps.getParameter< int >("filterTriggerType"))
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter < bool > ("verbose", false);

  if (verbose_)
    edm::LogInfo("L1TGCT") << "L1TGCT: constructor...." << std::endl;

  outputFile_ = ps.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    edm::LogInfo("L1TGCT") << "L1T Monitoring histograms will be saved to "
                           << outputFile_ << std::endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }

  //set Token(-s)
  gctIsoEmSourceToken_ = consumes<L1GctEmCandCollection>(ps.getParameter<edm::InputTag>("gctIsoEmSource"));
  gctNonIsoEmSourceToken_ = consumes<L1GctEmCandCollection>(ps.getParameter<edm::InputTag>("gctNonIsoEmSource"));
  gctCenJetsSourceToken_ = consumes<L1GctJetCandCollection>(ps.getParameter<edm::InputTag>("gctCentralJetsSource"));
  gctForJetsSourceToken_ = consumes<L1GctJetCandCollection>(ps.getParameter<edm::InputTag>("gctForwardJetsSource"));
  gctTauJetsSourceToken_ = consumes<L1GctJetCandCollection>(ps.getParameter<edm::InputTag>("gctTauJetsSource"));
  if(m_stage1_layer2_ == true){
      gctIsoTauJetsSourceToken_=consumes<L1GctJetCandCollection>(ps.getParameter<edm::InputTag>("gctIsoTauJetsSource"));
  }
  gctEnergySumsSourceToken_ = consumes<L1GctHFRingEtSumsCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsSource"));
  l1HFCountsToken_ = consumes<L1GctHFBitCountsCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsSource"));
  l1EtMissToken_ = consumes<L1GctEtMissCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsSource"));
  l1HtMissToken_ = consumes<L1GctHtMissCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsSource"));
  l1EtHadToken_ = consumes<L1GctEtHadCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsSource"));
  l1EtTotalToken_ = consumes<L1GctEtTotalCollection>(ps.getParameter<edm::InputTag>("gctEnergySumsSource"));
}

L1TGCT::~L1TGCT()
{
}

void L1TGCT::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&)
{

  nev_ = 0;

  ibooker.setCurrentFolder(monitorDir_);

  runId_     = ibooker.bookInt("iRun");
  runId_->Fill(-1);
  lumisecId_ = ibooker.bookInt("iLumiSection");
  lumisecId_->Fill(-1);
  
  triggerType_ =ibooker.book1D("TriggerType", "TriggerType", 17, -0.5, 16.5);

  l1GctAllJetsEtEtaPhi_ = ibooker.book2D("AllJetsEtEtaPhi", "CENTRAL AND FORWARD JET E_{T}",JETETABINS, JETETAMIN, JETETAMAX,PHIBINS, PHIMIN, PHIMAX);
  l1GctCenJetsEtEtaPhi_ = ibooker.book2D("CenJetsEtEtaPhi", "CENTRAL JET E_{T}",JETETABINS, JETETAMIN, JETETAMAX, PHIBINS, PHIMIN, PHIMAX); 
  l1GctForJetsEtEtaPhi_ = ibooker.book2D("ForJetsEtEtaPhi", "FORWARD JET E_{T}", JETETABINS, JETETAMIN, JETETAMAX, PHIBINS, PHIMIN, PHIMAX); 
  l1GctTauJetsEtEtaPhi_ = ibooker.book2D("TauJetsEtEtaPhi", "TAU JET E_{T}", EMETABINS, EMETAMIN, EMETAMAX,	PHIBINS, PHIMIN, PHIMAX);
  if (m_stage1_layer2_ == true){
    l1GctIsoTauJetsEtEtaPhi_ = ibooker.book2D("IsoTauJetsEtEtaPhi", "ISOTAU JET E_{T}", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX);
  }
  l1GctIsoEmRankEtaPhi_ = ibooker.book2D("IsoEmRankEtaPhi", "ISO EM E_{T}", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX); 		    
  l1GctNonIsoEmRankEtaPhi_ = ibooker.book2D("NonIsoEmRankEtaPhi", "NON-ISO EM E_{T}", EMETABINS, EMETAMIN, EMETAMAX,PHIBINS, PHIMIN, PHIMAX); 
  l1GctAllJetsOccEtaPhi_ = ibooker.book2D("AllJetsOccEtaPhi", "CENTRAL AND FORWARD JET OCCUPANCY", JETETABINS, JETETAMIN, JETETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctCenJetsOccEtaPhi_ = ibooker.book2D("CenJetsOccEtaPhi", "CENTRAL JET OCCUPANCY", JETETABINS, JETETAMIN, JETETAMAX, PHIBINS, PHIMIN, PHIMAX); 
  l1GctForJetsOccEtaPhi_ = ibooker.book2D("ForJetsOccEtaPhi", "FORWARD JET OCCUPANCY",JETETABINS, JETETAMIN, JETETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctTauJetsOccEtaPhi_ = ibooker.book2D("TauJetsOccEtaPhi", "TAU JET OCCUPANCY", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX);
  if (m_stage1_layer2_ == true){
    l1GctIsoTauJetsOccEtaPhi_ = ibooker.book2D("IsoTauJetsOccEtaPhi", "ISOTAU JET OCCUPANCY", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX);
  }
  l1GctIsoEmOccEtaPhi_ = ibooker.book2D("IsoEmOccEtaPhi", "ISO EM OCCUPANCY", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX);
  l1GctNonIsoEmOccEtaPhi_ = ibooker.book2D("NonIsoEmOccEtaPhi", "NON-ISO EM OCCUPANCY", EMETABINS, EMETAMIN, EMETAMAX, PHIBINS, PHIMIN, PHIMAX); 
  

    //HF Ring stuff

  l1GctHFRing1TowerCountPosEtaNegEta_ = ibooker.book2D("HFRing1TowerCountCorr", "HF RING1 TOWER COUNT CORRELATION +/-  #eta", R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX);
  l1GctHFRing2TowerCountPosEtaNegEta_ = ibooker.book2D("HFRing2TowerCountCorr", "HF RING2 TOWER COUNT CORRELATION +/-  #eta", R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX);
  
  l1GctHFRing1TowerCountPosEta_ = ibooker.book1D("HFRing1TowerCountPosEta", "HF RING1 TOWER COUNT  #eta  +", R3BINS, R3MIN, R3MAX);
  l1GctHFRing1TowerCountNegEta_ = ibooker.book1D("HFRing1TowerCountNegEta", "HF RING1 TOWER COUNT  #eta  -", R3BINS, R3MIN, R3MAX);
  l1GctHFRing2TowerCountPosEta_ = ibooker.book1D("HFRing2TowerCountPosEta", "HF RING2 TOWER COUNT  #eta  +", R3BINS, R3MIN, R3MAX);
  l1GctHFRing2TowerCountNegEta_ = ibooker.book1D("HFRing2TowerCountNegEta", "HF RING2 TOWER COUNT  #eta  -", R3BINS, R3MIN, R3MAX);

  l1GctHFRingTowerCountOccBx_ = ibooker.book2D("HFRingTowerCountOccBx", "HF RING TOWER COUNT PER BX",BXBINS, BXMIN, BXMAX, R3BINS, R3MIN, R3MAX);
  
  if (m_stage1_layer2_ == false){
    l1GctHFRing1PosEtaNegEta_ = ibooker.book2D("HFRing1Corr", "HF RING1 E_{T} CORRELATION +/-  #eta",  R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX); 
    l1GctHFRing2PosEtaNegEta_ = ibooker.book2D("HFRing2Corr", "HF RING2 E_{T} CORRELATION +/-  #eta", R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX);
    l1GctHFRing1ETSumPosEta_ = ibooker.book1D("HFRing1ETSumPosEta", "HF RING1 E_{T}  #eta  +", R3BINS, R3MIN, R3MAX);
    l1GctHFRing1ETSumNegEta_ = ibooker.book1D("HFRing1ETSumNegEta", "HF RING1 E_{T}  #eta  -", R3BINS, R3MIN, R3MAX);
    l1GctHFRing2ETSumPosEta_ = ibooker.book1D("HFRing2ETSumPosEta", "HF RING2 E_{T}  #eta  +", R3BINS, R3MIN, R3MAX);
    l1GctHFRing2ETSumNegEta_ = ibooker.book1D("HFRing2ETSumNegEta", "HF RING2 E_{T}  #eta  -", R3BINS, R3MIN, R3MAX);
    l1GctHFRingETSumOccBx_ = ibooker.book2D("HFRingETSumOccBx", "HF RING E_{T} PER BX",BXBINS, BXMIN, BXMAX, R3BINS, R3MIN, R3MAX);
    l1GctHFRingRatioPosEta_  = ibooker.book1D("HFRingRatioPosEta", "HF RING E_{T} RATIO  #eta  +", R5BINS, R5MIN, R5MAX);
    l1GctHFRingRatioNegEta_  = ibooker.book1D("HFRingRatioNegEta", "HF RING E_{T} RATIO  #eta  -", R5BINS, R5MIN, R5MAX);
  }

  if (m_stage1_layer2_ == true){
    l1GctHFRing1PosEtaNegEta_ = ibooker.book2D("IsoTau 1 2 Corr", "IsoTau 1 IsoTau 2 E_{T} CORRELATION",  R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX); 
    l1GctHFRing2PosEtaNegEta_ = ibooker.book2D("IsoTau 3 4 Corr", "IsoTau 3 IsoTau 4 CORRELATION", R3BINS, R3MIN, R3MAX, R3BINS, R3MIN, R3MAX);
    l1GctHFRing1ETSumPosEta_ = ibooker.book1D("Iso Tau 1 Et", "Isolated Tau1 E_{T}", 9, -0.5, 8.5);
    l1GctHFRing1ETSumNegEta_ = ibooker.book1D("Iso Tau 2 Et", "Isolated Tau2 E_{T}", 9, -0.5, 8.5);
    l1GctHFRing2ETSumPosEta_ = ibooker.book1D("Iso Tau 3 Et", "Isolated Tau3 E_{T}", 9, -0.5, 8.5);
    l1GctHFRing2ETSumNegEta_ = ibooker.book1D("Iso Tau 4 Et", "Isolated Tau4 E_{T}", 9, -0.5, 8.5);
    l1GctHFRingETSumOccBx_ = ibooker.book2D("IsoTau HFRingSum OccBx", "Iso Tau PER BX",BXBINS, BXMIN, BXMAX, R3BINS, R3MIN, R3MAX);
    l1GctHFRingRatioPosEta_  = ibooker.book1D("IsoTau Ratio 1 2", "IsoTau E_{T} RATIO", 9, -0.5, 8.5);
    l1GctHFRingRatioNegEta_  = ibooker.book1D("IsoTau Ratio 1 2", "IsoTau E_{T} RATIO", 9, -0.5, 8.5);
  }
    
    // Rank histograms
  l1GctCenJetsRank_  = ibooker.book1D("CenJetsRank", "CENTRAL JET E_{T}", R6BINS, R6MIN, R6MAX);
  l1GctForJetsRank_  = ibooker.book1D("ForJetsRank", "FORWARD JET E_{T}", R6BINS, R6MIN, R6MAX);
  l1GctTauJetsRank_  = ibooker.book1D("TauJetsRank", "TAU JET E_{T}", R6BINS, R6MIN, R6MAX);
  if (m_stage1_layer2_ == true){
    l1GctIsoTauJetsRank_ = ibooker.book1D("IsoTauJetsRank", "ISOTAU JET E_{T}", R6BINS, R6MIN, R6MAX);
  }
  l1GctIsoEmRank_    = ibooker.book1D("IsoEmRank", "ISO EM E_{T}", R6BINS, R6MIN, R6MAX);
  l1GctNonIsoEmRank_ = ibooker.book1D("NonIsoEmRank", "NON-ISO EM E_{T}", R6BINS, R6MIN, R6MAX);

  l1GctAllJetsOccRankBx_ = ibooker.book2D("AllJetsOccRankBx","ALL JETS E_{T} PER BX",BXBINS,BXMIN,BXMAX,R6BINS,R6MIN,R6MAX);
  l1GctAllEmOccRankBx_   = ibooker.book2D("AllEmOccRankBx","ALL EM E_{T} PER BX",BXBINS,BXMIN,BXMAX,R6BINS,R6MIN,R6MAX);

    // Energy sums
  l1GctEtMiss_    = ibooker.book1D("EtMiss", "MET", R12BINS, R12MIN, R12MAX);
  l1GctEtMissPhi_ = ibooker.book1D("EtMissPhi", "MET  #phi", METPHIBINS, METPHIMIN, METPHIMAX);
  l1GctEtMissOf_  = ibooker.book1D("EtMissOf", "MET OVERFLOW", OFBINS, OFMIN, OFMAX);
  l1GctEtMissOccBx_ = ibooker.book2D("EtMissOccBx","MET PER BX",BXBINS,BXMIN,BXMAX,R12BINS,R12MIN,R12MAX);
  if (m_stage1_layer2_ == false) {
    l1GctHtMiss_    = ibooker.book1D("HtMiss", "MHT", R7BINS, R7MIN, R7MAX);
    l1GctHtMissPhi_ = ibooker.book1D("HtMissPhi", "MHT  #phi", MHTPHIBINS, MHTPHIMIN, MHTPHIMAX);
    l1GctHtMissOf_  = ibooker.book1D("HtMissOf", "MHT OVERFLOW", OFBINS, OFMIN, OFMAX);
    l1GctHtMissOccBx_ = ibooker.book2D("HtMissOccBx","MHT PER BX",BXBINS,BXMIN,BXMAX,R7BINS,R7MIN,R7MAX);
  }
  if (m_stage1_layer2_ == true) {   
    l1GctHtMiss_    = ibooker.book1D("HtMissHtTotal", "MHTHT", R7BINS, R7MIN, R7MAX);
    l1GctHtMissPhi_ = ibooker.book1D("HtMissHtTotal Phi", "MHTHT  #phi", MHTPHIBINS, MHTPHIMIN, MHTPHIMAX);
    l1GctHtMissOf_  = ibooker.book1D("HtMissHtTotal Of", "MHTHT OVERFLOW", OFBINS, OFMIN, OFMAX);
    l1GctHtMissOccBx_ = ibooker.book2D("HtMissHtTotal OccBx","MHTHT PER BX",BXBINS,BXMIN,BXMAX,R7BINS,R7MIN,R7MAX);
  }
  l1GctEtMissHtMissCorr_ = ibooker.book2D("EtMissHtMissCorr", "MET MHT CORRELATION", R6BINS, R12MIN, R12MAX, R6BINS, R7MIN, R7MAX); 
  l1GctEtMissHtMissCorrPhi_ = ibooker.book2D("EtMissHtMissPhiCorr", "MET MHT  #phi  CORRELATION", METPHIBINS, METPHIMIN, METPHIMAX, MHTPHIBINS, MHTPHIMIN, MHTPHIMAX);
  l1GctEtTotal_   = ibooker.book1D("EtTotal", "SUM E_{T}", R12BINS, R12MIN, R12MAX);
  l1GctEtTotalOf_ = ibooker.book1D("EtTotalOf", "SUM E_{T} OVERFLOW", OFBINS, OFMIN, OFMAX);
  l1GctEtTotalOccBx_ = ibooker.book2D("EtTotalOccBx","SUM E_{T} PER BX",BXBINS,BXMIN,BXMAX,R12BINS,R12MIN,R12MAX);
  l1GctEtHad_     = ibooker.book1D("EtHad", "H_{T}", R12BINS, R12MIN, R12MAX);
  l1GctEtHadOf_   = ibooker.book1D("EtHadOf", "H_{T} OVERFLOW", OFBINS, OFMIN, OFMAX);
  l1GctEtHadOccBx_ = ibooker.book2D("EtHadOccBx","H_{T} PER BX",BXBINS,BXMIN,BXMAX,R12BINS,R12MIN,R12MAX);
  l1GctEtTotalEtHadCorr_ = ibooker.book2D("EtTotalEtHadCorr", "Sum E_{T} H_{T} CORRELATION", R6BINS, R12MIN, R12MAX, R6BINS, R12MIN, R12MAX); 
  //}
}


void L1TGCT::dqmBeginRun(edm::Run const& iRrun, edm::EventSetup const& evSetup) {
  //runId_->Fill(iRrun.id().run());
}

void L1TGCT::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& evSetup) {
  //lumisecId_->Fill(iLumi.id().luminosityBlock());
}

void L1TGCT::analyze(const edm::Event & e, const edm::EventSetup & c)
{
  nev_++;
  if (verbose_) {
    edm::LogInfo("L1TGCT") << "L1TGCT: analyze...." << std::endl;
  }

  
  // filter according trigger type
  //  enum ExperimentType {
  //        Undefined          =  0,
  //        PhysicsTrigger     =  1,
  //        CalibrationTrigger =  2,
  //        RandomTrigger      =  3,
  //        Reserved           =  4,
  //        TracedEvent        =  5,
  //        TestTrigger        =  6,
  //        ErrorTrigger       = 15

  // fill a histogram with the trigger type, for normalization fill also last bin
  // ErrorTrigger + 1
  double triggerType = static_cast<double> (e.experimentType()) + 0.001;
  double triggerTypeLast = static_cast<double> (edm::EventAuxiliary::ExperimentType::ErrorTrigger)
                          + 0.001;
  triggerType_->Fill(triggerType);
  triggerType_->Fill(triggerTypeLast + 1);

  // filter only if trigger type is greater than 0, negative values disable filtering
  if (filterTriggerType_ >= 0) {

      // now filter, for real data only
      if (e.isRealData()) {
          if (!(e.experimentType() == filterTriggerType_)) {

              edm::LogInfo("L1TGCT") << "\n Event of TriggerType "
                      << e.experimentType() << " rejected" << std::endl;
              return;

          }
      }

  }

  // Get all the collections
  edm::Handle < L1GctEmCandCollection > l1IsoEm;
  edm::Handle < L1GctEmCandCollection > l1NonIsoEm;
  edm::Handle < L1GctJetCandCollection > l1CenJets;
  edm::Handle < L1GctJetCandCollection > l1ForJets;
  edm::Handle < L1GctJetCandCollection > l1TauJets;
  if(m_stage1_layer2_ == true) {
    edm::Handle < L1GctJetCandCollection > l1IsoTauJets;
    e.getByToken(gctIsoTauJetsSourceToken_, l1IsoTauJets);
  }
  edm::Handle < L1GctHFRingEtSumsCollection > l1HFSums; 
  edm::Handle < L1GctHFBitCountsCollection > l1HFCounts;
  edm::Handle < L1GctEtMissCollection >  l1EtMiss;
  edm::Handle < L1GctHtMissCollection >  l1HtMiss;
  edm::Handle < L1GctEtHadCollection >   l1EtHad;
  edm::Handle < L1GctEtTotalCollection > l1EtTotal;

  e.getByToken(gctIsoEmSourceToken_, l1IsoEm);
  e.getByToken(gctNonIsoEmSourceToken_, l1NonIsoEm);
  e.getByToken(gctCenJetsSourceToken_, l1CenJets);
  e.getByToken(gctForJetsSourceToken_, l1ForJets);
  e.getByToken(gctTauJetsSourceToken_, l1TauJets);
  e.getByToken(gctEnergySumsSourceToken_, l1HFSums);
  e.getByToken(l1HFCountsToken_, l1HFCounts);
  e.getByToken(l1EtMissToken_, l1EtMiss);
  e.getByToken(l1HtMissToken_, l1HtMiss);
  e.getByToken(l1EtHadToken_, l1EtHad);
  e.getByToken(l1EtTotalToken_, l1EtTotal);

  // Fill histograms

  // Central jets
  if (l1CenJets.isValid()) {
    for (L1GctJetCandCollection::const_iterator cj = l1CenJets->begin();cj != l1CenJets->end(); cj++) {
      // only plot central BX
      if (cj->bx()==0) {
        l1GctCenJetsRank_->Fill(cj->rank());
        // only plot eta and phi maps for non-zero candidates
        if (cj->rank()) {
          l1GctAllJetsEtEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi(),cj->rank());
          l1GctAllJetsOccEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi());
          l1GctCenJetsEtEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi(),cj->rank());
          l1GctCenJetsOccEtaPhi_->Fill(cj->regionId().ieta(),cj->regionId().iphi());
        }
      }
      if (cj->rank()) l1GctAllJetsOccRankBx_->Fill(cj->bx(),cj->rank()); // for all BX
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1CenJets label was " << gctCenJetsSource_ ;
  }

  // Forward jets
  if (l1ForJets.isValid()) {
    for (L1GctJetCandCollection::const_iterator fj = l1ForJets->begin(); fj != l1ForJets->end(); fj++) {
      // only plot central BX
      if (fj->bx()==0) {
        l1GctForJetsRank_->Fill(fj->rank());
        // only plot eta and phi maps for non-zero candidates
        if (fj->rank()) {
          l1GctAllJetsEtEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi(),fj->rank());
          l1GctAllJetsOccEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi());
          l1GctForJetsEtEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi(),fj->rank());
          l1GctForJetsOccEtaPhi_->Fill(fj->regionId().ieta(),fj->regionId().iphi());    
        }
      }
      if (fj->rank()) l1GctAllJetsOccRankBx_->Fill(fj->bx(),fj->rank()); // for all BX
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1ForJets label was " << gctForJetsSource_ ;
  }

  // Tau jets
  if (l1TauJets.isValid()) {
    for (L1GctJetCandCollection::const_iterator tj = l1TauJets->begin(); tj != l1TauJets->end(); tj++) {
      // only plot central BX
      if (tj->bx()==0) {
        l1GctTauJetsRank_->Fill(tj->rank());
        // only plot eta and phi maps for non-zero candidates
        if (tj->rank()) {
          l1GctTauJetsEtEtaPhi_->Fill(tj->regionId().ieta(),tj->regionId().iphi(),tj->rank());
          l1GctTauJetsOccEtaPhi_->Fill(tj->regionId().ieta(),tj->regionId().iphi());
        }
      }
      if (tj->rank()) l1GctAllJetsOccRankBx_->Fill(tj->bx(),tj->rank()); // for all BX
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1TauJets label was " << gctTauJetsSource_ ;
  }

   // IsoTau jets
  if (m_stage1_layer2_ == true) {
   edm::Handle < L1GctJetCandCollection > l1IsoTauJets;
   e.getByToken(gctIsoTauJetsSourceToken_, l1IsoTauJets);
   if (l1IsoTauJets.isValid()) {
    for (L1GctJetCandCollection::const_iterator itj = l1IsoTauJets->begin(); itj != l1IsoTauJets->end(); itj++) {
      // only plot central BX
      if (itj->bx()==0) {
        l1GctIsoTauJetsRank_->Fill(itj->rank());
        // only plot eta and phi maps for non-zero candidates
        if (itj->rank()) {
          l1GctIsoTauJetsEtEtaPhi_->Fill(itj->regionId().ieta(),itj->regionId().iphi(),itj->rank());
          l1GctIsoTauJetsOccEtaPhi_->Fill(itj->regionId().ieta(),itj->regionId().iphi());
        }
      }
      if (itj->rank()) l1GctAllJetsOccRankBx_->Fill(itj->bx(),itj->rank()); // for all BX
    }
   } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1IsoTauJets label was " << gctIsoTauJetsSource_ ;
   }
  }

  
  // Missing ET
  if (l1EtMiss.isValid()) { 
    for (L1GctEtMissCollection::const_iterator met = l1EtMiss->begin(); met != l1EtMiss->end(); met++) {
      // only plot central BX
      if (met->bx()==0) {
        if (met->overFlow() == 0 && met->et() > 0) {
          //Avoid problems with met=0 candidates affecting MET_PHI plots
          l1GctEtMiss_->Fill(met->et());
          l1GctEtMissPhi_->Fill(met->phi());
        }
        l1GctEtMissOf_->Fill(met->overFlow());
      }
      if (met->overFlow() == 0 && met->et() > 0) l1GctEtMissOccBx_->Fill(met->bx(),met->et()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtMiss label was " << gctEnergySumsSource_ ;    
  }

  // Missing HT
  if (l1HtMiss.isValid()) { 
    for (L1GctHtMissCollection::const_iterator mht = l1HtMiss->begin(); mht != l1HtMiss->end(); mht++) {
      // only plot central BX
      if (mht->bx()==0) {
        if (mht->overFlow() == 0 && mht->et() > 0) {
          //Avoid problems with mht=0 candidates affecting MHT_PHI plots
          l1GctHtMiss_->Fill(mht->et());
          l1GctHtMissPhi_->Fill(mht->phi());
        }
        l1GctHtMissOf_->Fill(mht->overFlow());
      }
      if (mht->overFlow() == 0 && mht->et() > 0) l1GctHtMissOccBx_->Fill(mht->bx(),mht->et()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1HtMiss label was " << gctEnergySumsSource_ ;    
  }

  // Missing ET HT correlations
  if (l1HtMiss.isValid() && l1EtMiss.isValid()) { 
    if (l1HtMiss->size() == l1EtMiss->size()) {
      for (unsigned i=0; i<l1HtMiss->size(); i++) {
        if (l1HtMiss->at(i).overFlow() == 0 && l1EtMiss->at(i).overFlow() == 0 && 
            l1HtMiss->at(i).bx() == 0 && l1EtMiss->at(i).bx() == 0) {
          // Avoid problems overflows and only plot central BX
          l1GctEtMissHtMissCorr_->Fill(l1EtMiss->at(i).et(),l1HtMiss->at(i).et());
          if (l1HtMiss->at(i).et() && l1EtMiss->at(i).et()){ // Don't plot phi if one or other is zero
            l1GctEtMissHtMissCorrPhi_->Fill(l1EtMiss->at(i).phi(),l1HtMiss->at(i).phi());
          }
        }
      }
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtMiss or l1HtMiss label was " << gctEnergySumsSource_ ;    
  }

  // HT 
  if (l1EtHad.isValid()) {
    for (L1GctEtHadCollection::const_iterator ht = l1EtHad->begin(); ht != l1EtHad->end(); ht++) {
      // only plot central BX
      if (ht->bx()==0) {
        l1GctEtHad_->Fill(ht->et());
        l1GctEtHadOf_->Fill(ht->overFlow());
      }
      l1GctEtHadOccBx_->Fill(ht->bx(),ht->et()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtHad label was " << gctEnergySumsSource_ ;    
  }

  // Total ET
  if (l1EtTotal.isValid()) {
    for (L1GctEtTotalCollection::const_iterator et = l1EtTotal->begin(); et != l1EtTotal->end(); et++) {
      // only plot central BX
      if (et->bx()==0) {
        l1GctEtTotal_->Fill(et->et());
        l1GctEtTotalOf_->Fill(et->overFlow());
      }
      l1GctEtTotalOccBx_->Fill(et->bx(),et->et()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtTotal label was " << gctEnergySumsSource_ ;    
  }

  // Total ET HT correlations
  if (l1EtTotal.isValid() && l1EtHad.isValid()) { 
    if (l1EtTotal->size() == l1EtHad->size()) {
      for (unsigned i=0; i<l1EtHad->size(); i++) {
        if (l1EtHad->at(i).overFlow() == 0 && l1EtTotal->at(i).overFlow() == 0 && 
            l1EtHad->at(i).bx() == 0 && l1EtTotal->at(i).bx() == 0) {
          // Avoid problems overflows and only plot central BX
          l1GctEtTotalEtHadCorr_->Fill(l1EtTotal->at(i).et(),l1EtHad->at(i).et());
        }
      }
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1EtTotal or l1EtHad label was " << gctEnergySumsSource_ ;    
  }

  //HF Ring Et Sums
  if (l1HFSums.isValid()) {
    for (L1GctHFRingEtSumsCollection::const_iterator hfs=l1HFSums->begin(); hfs!=l1HFSums->end(); hfs++){ 
      // only plot central BX
      if (hfs->bx()==0) {
        // Individual ring Et sums
        l1GctHFRing1ETSumPosEta_->Fill(hfs->etSum(0));
        l1GctHFRing1ETSumNegEta_->Fill(hfs->etSum(1));
        l1GctHFRing2ETSumPosEta_->Fill(hfs->etSum(2));
        l1GctHFRing2ETSumNegEta_->Fill(hfs->etSum(3));
        // Ratio of ring Et sums
        if (hfs->etSum(2)!=0) l1GctHFRingRatioPosEta_->Fill((hfs->etSum(0))/(hfs->etSum(2)));
        if (hfs->etSum(3)!=0) l1GctHFRingRatioNegEta_->Fill((hfs->etSum(1))/(hfs->etSum(3)));
        // Correlate positive and neagative eta
        l1GctHFRing1PosEtaNegEta_->Fill(hfs->etSum(0),hfs->etSum(1));
        l1GctHFRing2PosEtaNegEta_->Fill(hfs->etSum(2),hfs->etSum(3));
      }
      // Occupancy vs BX
      for (unsigned i=0; i<4; i++){
        l1GctHFRingETSumOccBx_->Fill(hfs->bx(),hfs->etSum(i));
      }
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1HFSums label was " << gctEnergySumsSource_ ;
  }

  // HF Ring Counts
  if (l1HFCounts.isValid()) {
    for (L1GctHFBitCountsCollection::const_iterator hfc=l1HFCounts->begin(); hfc!=l1HFCounts->end(); hfc++){ 
      // only plot central BX
      if (hfc->bx()==0) {
        // Individual ring counts
        l1GctHFRing1TowerCountPosEta_->Fill(hfc->bitCount(0));
        l1GctHFRing1TowerCountNegEta_->Fill(hfc->bitCount(1));
        l1GctHFRing2TowerCountPosEta_->Fill(hfc->bitCount(2));
        l1GctHFRing2TowerCountNegEta_->Fill(hfc->bitCount(3));
        // Correlate positive and negative eta
        l1GctHFRing1TowerCountPosEtaNegEta_->Fill(hfc->bitCount(0),hfc->bitCount(1));
        l1GctHFRing2TowerCountPosEtaNegEta_->Fill(hfc->bitCount(2),hfc->bitCount(3));
      }
      // Occupancy vs BX
      for (unsigned i=0; i<4; i++){
        l1GctHFRingTowerCountOccBx_->Fill(hfc->bx(),hfc->bitCount(i));
      }
    }
  } else {    
    edm::LogWarning("DataNotFound") << " Could not find l1HFCounts label was " << gctEnergySumsSource_ ;
  }
  
  // Isolated EM
  if (l1IsoEm.isValid()) {
    for (L1GctEmCandCollection::const_iterator ie=l1IsoEm->begin(); ie!=l1IsoEm->end(); ie++) {
      // only plot central BX
      if (ie->bx()==0) {
        l1GctIsoEmRank_->Fill(ie->rank());
        // only plot eta and phi maps for non-zero candidates
        if (ie->rank()){ 
          l1GctIsoEmRankEtaPhi_->Fill(ie->regionId().ieta(),ie->regionId().iphi(),ie->rank());
          l1GctIsoEmOccEtaPhi_->Fill(ie->regionId().ieta(),ie->regionId().iphi());
        }
      }
      if (ie->rank()) l1GctAllEmOccRankBx_->Fill(ie->bx(),ie->rank()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1IsoEm label was " << gctIsoEmSource_ ;
  } 

  // Non-isolated EM
  if (l1NonIsoEm.isValid()) { 
    for (L1GctEmCandCollection::const_iterator ne=l1NonIsoEm->begin(); ne!=l1NonIsoEm->end(); ne++) {
      // only plot central BX
      if (ne->bx()==0) {
        l1GctNonIsoEmRank_->Fill(ne->rank());
        // only plot eta and phi maps for non-zero candidates
        if (ne->rank()){ 
          l1GctNonIsoEmRankEtaPhi_->Fill(ne->regionId().ieta(),ne->regionId().iphi(),ne->rank());
          l1GctNonIsoEmOccEtaPhi_->Fill(ne->regionId().ieta(),ne->regionId().iphi());
        }
      }
      if (ne->rank()) l1GctAllEmOccRankBx_->Fill(ne->bx(),ne->rank()); // for all BX
    }
  } else {
    edm::LogWarning("DataNotFound") << " Could not find l1NonIsoEm label was " << gctNonIsoEmSource_ ;
  }
  edm::LogInfo("L1TGCT") << "L1TGCT: end job...." << std::endl;
  edm::LogInfo("EndJob") << "analyzed " << nev_ << " events";
}

  
