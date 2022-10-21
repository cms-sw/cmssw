/*
  HLTInclusiveVBFSource
  Phat Srimanobhas
  To monitor VBF DataParking
*/

#include "DQMOffline/Trigger/interface/HLTInclusiveVBFSource.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <cmath>
#include "TH1F.h"
#include "TProfile.h"
#include "TH2F.h"
#include "TPRegexp.h"
#include "TMath.h"

using namespace edm;
using namespace reco;
using namespace std;

HLTInclusiveVBFSource::HLTInclusiveVBFSource(const edm::ParameterSet& iConfig) {
  LogDebug("HLTInclusiveVBFSource") << "constructor....";
  nCount_ = 0;

  dirname_ = iConfig.getUntrackedParameter("dirname", std::string("HLT/InclusiveVBF"));
  processname_ = iConfig.getParameter<std::string>("processname");
  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
  triggerSummaryToken = consumes<trigger::TriggerEvent>(triggerSummaryLabel_);
  triggerResultsToken = consumes<edm::TriggerResults>(triggerResultsLabel_);
  triggerSummaryFUToken = consumes<trigger::TriggerEvent>(
      edm::InputTag(triggerSummaryLabel_.label(), triggerSummaryLabel_.instance(), std::string("FU")));
  triggerResultsFUToken = consumes<edm::TriggerResults>(
      edm::InputTag(triggerResultsLabel_.label(), triggerResultsLabel_.instance(), std::string("FU")));

  //path_                = iConfig.getUntrackedParameter<std::vector<std::string> >("paths");
  //l1path_              = iConfig.getUntrackedParameter<std::vector<std::string> >("l1paths");
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);

  caloJetsToken = consumes<reco::CaloJetCollection>(iConfig.getParameter<edm::InputTag>("CaloJetCollectionLabel"));
  caloMetToken = consumes<reco::CaloMETCollection>(iConfig.getParameter<edm::InputTag>("CaloMETCollectionLabel"));
  pfJetsToken = consumes<edm::View<reco::PFJet> >(iConfig.getParameter<edm::InputTag>("PFJetCollectionLabel"));
  pfMetToken = consumes<edm::View<reco::PFMET> >(iConfig.getParameter<edm::InputTag>("PFMETCollectionLabel"));
  //jetID                = new reco::helper::JetIDHelper(iConfig.getParameter<ParameterSet>("JetIDParams"));

  minPtHigh_ = iConfig.getUntrackedParameter<double>("minPtHigh", 40.);
  minPtLow_ = iConfig.getUntrackedParameter<double>("minPtLow", 40.);
  minDeltaEta_ = iConfig.getUntrackedParameter<double>("minDeltaEta", 3.5);
  deltaRMatch_ = iConfig.getUntrackedParameter<double>("deltaRMatch", 0.1);
  minInvMass_ = iConfig.getUntrackedParameter<double>("minInvMass", 1000.0);
  etaOpposite_ = iConfig.getUntrackedParameter<bool>("etaOpposite", true);

  check_mjj650_Pt35_DEta3p5 = false;
  check_mjj700_Pt35_DEta3p5 = false;
  check_mjj750_Pt35_DEta3p5 = false;
  check_mjj800_Pt35_DEta3p5 = false;
  check_mjj650_Pt40_DEta3p5 = false;
  check_mjj700_Pt40_DEta3p5 = false;
  check_mjj750_Pt40_DEta3p5 = false;
  check_mjj800_Pt40_DEta3p5 = false;
}

HLTInclusiveVBFSource::~HLTInclusiveVBFSource() {
  //
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void HLTInclusiveVBFSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace std;
  using namespace edm;
  using namespace trigger;
  using namespace reco;

  if (debug_)
    cout << "DEBUG-0: Start to analyze" << endl;

  //****************************************************
  // Get trigger information.
  //****************************************************
  //
  //---------- triggerResults ----------
  iEvent.getByToken(triggerResultsToken, triggerResults_);
  if (!triggerResults_.isValid()) {
    iEvent.getByToken(triggerResultsFUToken, triggerResults_);
    if (!triggerResults_.isValid()) {
      edm::LogInfo("HLTInclusiveVBFSource") << "TriggerResults not found, "
                                               "skipping event";
      return;
    }
  }

  // Check how many HLT triggers are in triggerResults
  triggerNames_ = iEvent.triggerNames(*triggerResults_);

  //---------- triggerSummary ----------
  iEvent.getByToken(triggerSummaryToken, triggerObj_);
  if (!triggerObj_.isValid()) {
    iEvent.getByToken(triggerSummaryFUToken, triggerObj_);
    if (!triggerObj_.isValid()) {
      edm::LogInfo("HLTInclusiveVBFSource") << "TriggerEvent not found, "
                                               "skipping event";
      return;
    }
  }

  if (debug_)
    cout << "DEBUG-1: Trigger information" << endl;

  //****************************************************
  // Get AOD information
  //****************************************************
  //
  edm::Handle<edm::View<reco::PFMET> > metSrc;
  bool ValidPFMET_ = iEvent.getByToken(pfMetToken, metSrc);
  if (!ValidPFMET_)
    return;

  edm::Handle<edm::View<reco::PFJet> > jetSrc;
  bool ValidPFJet_ = iEvent.getByToken(pfJetsToken, jetSrc);
  if (!ValidPFJet_)
    return;

  if (!metSrc.isValid())
    return;
  if (!jetSrc.isValid())
    return;
  const edm::View<reco::PFMET>& mets = *metSrc;
  const edm::View<reco::PFJet>& jets = *jetSrc;
  if (jets.empty())
    return;
  if (mets.empty())
    return;

  if (debug_)
    cout << "DEBUG-2: AOD Information" << endl;

  //****************************************************
  // Variable setting
  //****************************************************
  //
  pathname = "dummy";
  filtername = "dummy";

  //
  reco_ejet1 = 0.;
  //reco_etjet1               = 0.;
  reco_pxjet1 = 0.;
  reco_pyjet1 = 0.;
  reco_pzjet1 = 0.;
  reco_ptjet1 = 0.;
  reco_etajet1 = 0.;
  reco_phijet1 = 0.;

  //
  reco_ejet2 = 0.;
  //reco_etjet2               = 0.;
  reco_pxjet2 = 0.;
  reco_pyjet2 = 0.;
  reco_pzjet2 = 0.;
  reco_ptjet2 = 0.;
  reco_etajet2 = 0.;
  reco_phijet2 = 0.;

  //
  hlt_ejet1 = 0.;
  //hlt_etjet1                = 0.;
  hlt_pxjet1 = 0.;
  hlt_pyjet1 = 0.;
  hlt_pzjet1 = 0.;
  hlt_ptjet1 = 0.;
  hlt_etajet1 = 0.;
  hlt_phijet1 = 0.;

  //
  hlt_ejet2 = 0.;
  //hlt_etjet2                = 0.;
  hlt_pxjet2 = 0.;
  hlt_pyjet2 = 0.;
  hlt_pzjet2 = 0.;
  hlt_ptjet2 = 0.;
  hlt_etajet2 = 0.;
  hlt_phijet2 = 0.;

  //
  checkOffline = false;
  checkHLT = false;
  checkHLTIndex = false;

  //
  dR_HLT_RECO_11 = 0.;
  dR_HLT_RECO_22 = 0.;
  dR_HLT_RECO_12 = 0.;
  dR_HLT_RECO_21 = 0.;

  //
  checkdR_sameOrder = false;
  checkdR_crossOrder = false;

  //
  reco_deltaetajet = 0.;
  reco_deltaphijet = 0.;
  reco_invmassjet = 0.;
  hlt_deltaetajet = 0.;
  hlt_deltaphijet = 0.;
  hlt_invmassjet = 0.;

  //****************************************************
  // Offline analysis
  //****************************************************
  //
  checkOffline = false;
  for (unsigned int ijet1 = 0; ijet1 < jets.size(); ijet1++) {
    if (jets[ijet1].neutralHadronEnergyFraction() > 0.99)
      continue;
    if (jets[ijet1].neutralEmEnergyFraction() > 0.99)
      continue;
    for (unsigned int ijet2 = ijet1 + 1; ijet2 < jets.size(); ijet2++) {
      if (jets[ijet2].neutralHadronEnergyFraction() > 0.99)
        continue;
      if (jets[ijet2].neutralEmEnergyFraction() > 0.99)
        continue;
      //
      reco_ejet1 = jets[ijet1].energy();
      //reco_etjet1  = jets[ijet1].et();
      reco_pxjet1 = jets[ijet1].momentum().X();
      reco_pyjet1 = jets[ijet1].momentum().Y();
      reco_pzjet1 = jets[ijet1].momentum().Z();
      reco_ptjet1 = jets[ijet1].pt();
      reco_etajet1 = jets[ijet1].eta();
      reco_phijet1 = jets[ijet1].phi();
      //
      reco_ejet2 = jets[ijet2].energy();
      //reco_etjet2  = jets[ijet2].et();
      reco_pxjet2 = jets[ijet2].momentum().X();
      reco_pyjet2 = jets[ijet2].momentum().Y();
      reco_pzjet2 = jets[ijet2].momentum().Z();
      reco_ptjet2 = jets[ijet2].pt();
      reco_etajet2 = jets[ijet2].eta();
      reco_phijet2 = jets[ijet2].phi();
      //
      reco_deltaetajet = reco_etajet1 - reco_etajet2;
      reco_deltaphijet = reco::deltaPhi(reco_phijet1, reco_phijet2);
      reco_invmassjet = sqrt((reco_ejet1 + reco_ejet2) * (reco_ejet1 + reco_ejet2) -
                             (reco_pxjet1 + reco_pxjet2) * (reco_pxjet1 + reco_pxjet2) -
                             (reco_pyjet1 + reco_pyjet2) * (reco_pyjet1 + reco_pyjet2) -
                             (reco_pzjet1 + reco_pzjet2) * (reco_pzjet1 + reco_pzjet2));

      //
      if (reco_ptjet1 < minPtHigh_)
        continue;
      if (reco_ptjet2 < minPtLow_)
        continue;
      if (etaOpposite_ == true && reco_etajet1 * reco_etajet2 > 0)
        continue;
      if (std::abs(reco_deltaetajet) < minDeltaEta_)
        continue;
      if (std::abs(reco_invmassjet) < minInvMass_)
        continue;

      //
      if (debug_)
        cout << "DEBUG-3" << endl;
      checkOffline = true;
      break;
    }
    if (checkOffline == true)
      break;
  }
  if (checkOffline == false)
    return;

  //****************************************************
  // Trigger efficiency: Loop for all VBF paths
  //****************************************************
  //const unsigned int numberOfPaths(hltConfig_.size());
  const trigger::TriggerObjectCollection& toc(triggerObj_->getObjects());
  for (auto& v : hltPathsAll_) {
    checkHLT = false;
    checkHLTIndex = false;

    //
    v.getMEhisto_RECO_deltaEta_DiJet()->Fill(reco_deltaetajet);
    v.getMEhisto_RECO_deltaPhi_DiJet()->Fill(reco_deltaphijet);
    v.getMEhisto_RECO_invMass_DiJet()->Fill(reco_invmassjet);

    //
    if (debug_)
      cout << "DEBUG-4-0: Path loops" << endl;

    //
    if (isHLTPathAccepted(v.getPath()) == false)
      continue;
    checkHLT = true;

    //
    if (debug_)
      cout << "DEBUG-4-1: Path is accepted. Now we are looking for " << v.getLabel() << " module." << endl;

    //
    edm::InputTag hltTag(v.getLabel(), "", processname_);
    const int hltIndex = triggerObj_->filterIndex(hltTag);
    if (hltIndex >= triggerObj_->sizeFilters())
      continue;
    checkHLT = true;
    if (debug_)
      cout << "DEBUG-4-2: HLT module " << v.getLabel() << " exists" << endl;
    const trigger::Keys& khlt = triggerObj_->filterKeys(hltIndex);
    auto kj = khlt.begin();
    for (; kj != khlt.end(); kj += 2) {
      if (debug_)
        cout << "DEBUG-5" << endl;
      checkdR_sameOrder = false;
      checkdR_crossOrder = false;  //
      hlt_ejet1 = toc[*kj].energy();
      //hlt_etjet1  = toc[*kj].et();
      hlt_pxjet1 = toc[*kj].px();
      hlt_pyjet1 = toc[*kj].py();
      hlt_pzjet1 = toc[*kj].pz();
      hlt_ptjet1 = toc[*kj].pt();
      hlt_etajet1 = toc[*kj].eta();
      hlt_phijet1 = toc[*kj].phi();
      //
      hlt_ejet2 = toc[*(kj + 1)].energy();
      //hlt_etjet2  = toc[*(kj+1)].et();
      hlt_pxjet2 = toc[*(kj + 1)].px();
      hlt_pyjet2 = toc[*(kj + 1)].py();
      hlt_pzjet2 = toc[*(kj + 1)].pz();
      hlt_ptjet2 = toc[*(kj + 1)].pt();
      hlt_etajet2 = toc[*(kj + 1)].eta();
      hlt_phijet2 = toc[*(kj + 1)].phi();
      //
      dR_HLT_RECO_11 = reco::deltaR(hlt_etajet1, hlt_phijet1, reco_etajet1, reco_phijet1);
      dR_HLT_RECO_22 = reco::deltaR(hlt_etajet2, hlt_phijet2, reco_etajet2, reco_phijet2);
      dR_HLT_RECO_12 = reco::deltaR(hlt_etajet1, hlt_phijet1, reco_etajet2, reco_phijet2);
      dR_HLT_RECO_21 = reco::deltaR(hlt_etajet2, hlt_phijet2, reco_etajet1, reco_phijet1);
      if (dR_HLT_RECO_11 < deltaRMatch_ && dR_HLT_RECO_22 < deltaRMatch_)
        checkdR_sameOrder = true;
      if (dR_HLT_RECO_12 < deltaRMatch_ && dR_HLT_RECO_21 < deltaRMatch_)
        checkdR_crossOrder = true;
      if (checkdR_sameOrder == false && checkdR_crossOrder == false)
        continue;
      checkHLTIndex = true;
      //
      if (debug_)
        cout << "DEBUG-6: Match" << endl;
      hlt_deltaetajet = hlt_etajet1 - hlt_etajet2;
      hlt_deltaphijet = reco::deltaPhi(hlt_phijet1, hlt_phijet2);
      if (checkdR_crossOrder) {
        hlt_deltaetajet = (-1) * hlt_deltaetajet;
        hlt_deltaphijet = reco::deltaPhi(hlt_phijet2, hlt_phijet1);
      }
      hlt_invmassjet = sqrt((hlt_ejet1 + hlt_ejet2) * (hlt_ejet1 + hlt_ejet2) -
                            (hlt_pxjet1 + hlt_pxjet2) * (hlt_pxjet1 + hlt_pxjet2) -
                            (hlt_pyjet1 + hlt_pyjet2) * (hlt_pyjet1 + hlt_pyjet2) -
                            (hlt_pzjet1 + hlt_pzjet2) * (hlt_pzjet1 + hlt_pzjet2));
      v.getMEhisto_HLT_deltaEta_DiJet()->Fill(hlt_deltaetajet);
      v.getMEhisto_HLT_deltaPhi_DiJet()->Fill(hlt_deltaphijet);
      v.getMEhisto_HLT_invMass_DiJet()->Fill(hlt_invmassjet);
      //
      v.getMEhisto_RECO_deltaEta_DiJet_Match()->Fill(reco_deltaetajet);
      v.getMEhisto_RECO_deltaPhi_DiJet_Match()->Fill(reco_deltaphijet);
      v.getMEhisto_RECO_invMass_DiJet_Match()->Fill(reco_invmassjet);
      //
      v.getMEhisto_RECOHLT_deltaEta()->Fill(reco_deltaetajet, hlt_deltaetajet);
      v.getMEhisto_RECOHLT_deltaPhi()->Fill(reco_deltaphijet, hlt_deltaphijet);
      v.getMEhisto_RECOHLT_invMass()->Fill(reco_invmassjet, hlt_invmassjet);
      //
      if (checkHLTIndex == true)
        break;
    }

    //****************************************************
    // Match information
    //****************************************************
    if (checkHLT == true && checkHLTIndex == true) {
      if (debug_)
        cout << "DEBUG-7: Match" << endl;
      v.getMEhisto_NumberOfMatches()->Fill(1);
    } else {
      if (debug_)
        cout << "DEBUG-8: Not match" << endl;
      v.getMEhisto_NumberOfMatches()->Fill(0);
    }
  }

  //****************************************************
  //
  //****************************************************
  for (auto& v : hltPathsAll_) {
    if (isHLTPathAccepted(v.getPath()) == false)
      continue;
    if (debug_)
      cout << "DEBUG-9: Loop for rate approximation: " << v.getPath() << endl;
    check_mjj650_Pt35_DEta3p5 = false;
    check_mjj700_Pt35_DEta3p5 = false;
    check_mjj750_Pt35_DEta3p5 = false;
    check_mjj800_Pt35_DEta3p5 = false;
    check_mjj650_Pt40_DEta3p5 = false;
    check_mjj700_Pt40_DEta3p5 = false;
    check_mjj750_Pt40_DEta3p5 = false;
    check_mjj800_Pt40_DEta3p5 = false;
    edm::InputTag hltTag(v.getLabel(), "", processname_);
    const int hltIndex = triggerObj_->filterIndex(hltTag);
    if (hltIndex >= triggerObj_->sizeFilters())
      continue;
    const trigger::Keys& khlt = triggerObj_->filterKeys(hltIndex);
    auto kj = khlt.begin();
    for (; kj != khlt.end(); kj += 2) {
      checkdR_sameOrder = false;
      checkdR_crossOrder = false;
      //
      hlt_ejet1 = toc[*kj].energy();
      //hlt_etjet1  = toc[*kj].et();
      hlt_pxjet1 = toc[*kj].px();
      hlt_pyjet1 = toc[*kj].py();
      hlt_pzjet1 = toc[*kj].pz();
      hlt_ptjet1 = toc[*kj].pt();
      hlt_etajet1 = toc[*kj].eta();
      hlt_phijet1 = toc[*kj].phi();
      //
      hlt_ejet2 = toc[*(kj + 1)].energy();
      //hlt_etjet2  = toc[*(kj+1)].et();
      hlt_pxjet2 = toc[*(kj + 1)].px();
      hlt_pyjet2 = toc[*(kj + 1)].py();
      hlt_pzjet2 = toc[*(kj + 1)].pz();
      hlt_ptjet2 = toc[*(kj + 1)].pt();
      hlt_etajet2 = toc[*(kj + 1)].eta();
      hlt_phijet2 = toc[*(kj + 1)].phi();
      //
      hlt_deltaetajet = hlt_etajet1 - hlt_etajet2;
      hlt_deltaphijet = reco::deltaPhi(hlt_phijet1, hlt_phijet2);
      hlt_invmassjet = sqrt((hlt_ejet1 + hlt_ejet2) * (hlt_ejet1 + hlt_ejet2) -
                            (hlt_pxjet1 + hlt_pxjet2) * (hlt_pxjet1 + hlt_pxjet2) -
                            (hlt_pyjet1 + hlt_pyjet2) * (hlt_pyjet1 + hlt_pyjet2) -
                            (hlt_pzjet1 + hlt_pzjet2) * (hlt_pzjet1 + hlt_pzjet2));
      //
      if (check_mjj650_Pt35_DEta3p5 == false && hlt_ptjet1 > 35. && hlt_ptjet2 >= 35. && hlt_invmassjet > 650 &&
          std::abs(hlt_deltaetajet) > 3.5) {
        check_mjj650_Pt35_DEta3p5 = true;
      }
      if (check_mjj700_Pt35_DEta3p5 == false && hlt_ptjet1 > 35. && hlt_ptjet2 >= 35. && hlt_invmassjet > 700 &&
          std::abs(hlt_deltaetajet) > 3.5) {
        check_mjj700_Pt35_DEta3p5 = true;
      }
      if (check_mjj750_Pt35_DEta3p5 == false && hlt_ptjet1 > 35. && hlt_ptjet2 >= 35. && hlt_invmassjet > 750 &&
          std::abs(hlt_deltaetajet) > 3.5) {
        check_mjj750_Pt35_DEta3p5 = true;
      }
      if (check_mjj800_Pt35_DEta3p5 == false && hlt_ptjet1 > 35. && hlt_ptjet2 >= 35. && hlt_invmassjet > 800 &&
          std::abs(hlt_deltaetajet) > 3.5) {
        check_mjj800_Pt35_DEta3p5 = true;
      }
      if (check_mjj650_Pt40_DEta3p5 == false && hlt_ptjet1 > 40. && hlt_ptjet2 >= 40. && hlt_invmassjet > 650 &&
          std::abs(hlt_deltaetajet) > 3.5) {
        check_mjj650_Pt40_DEta3p5 = true;
      }
      if (check_mjj700_Pt40_DEta3p5 == false && hlt_ptjet1 > 40. && hlt_ptjet2 >= 40. && hlt_invmassjet > 700 &&
          std::abs(hlt_deltaetajet) > 3.5) {
        check_mjj700_Pt40_DEta3p5 = true;
      }
      if (check_mjj750_Pt40_DEta3p5 == false && hlt_ptjet1 > 40. && hlt_ptjet2 >= 40. && hlt_invmassjet > 750 &&
          std::abs(hlt_deltaetajet) > 3.5) {
        check_mjj750_Pt40_DEta3p5 = true;
      }
      if (check_mjj800_Pt40_DEta3p5 == false && hlt_ptjet1 > 40. && hlt_ptjet2 >= 40. && hlt_invmassjet > 800 &&
          std::abs(hlt_deltaetajet) > 3.5) {
        check_mjj800_Pt40_DEta3p5 = true;
      }
    }
    if (check_mjj650_Pt35_DEta3p5 == true)
      v.getMEhisto_NumberOfEvents()->Fill(0);
    if (check_mjj700_Pt35_DEta3p5 == true)
      v.getMEhisto_NumberOfEvents()->Fill(1);
    if (check_mjj750_Pt35_DEta3p5 == true)
      v.getMEhisto_NumberOfEvents()->Fill(2);
    if (check_mjj800_Pt35_DEta3p5 == true)
      v.getMEhisto_NumberOfEvents()->Fill(3);
    if (check_mjj650_Pt40_DEta3p5 == true)
      v.getMEhisto_NumberOfEvents()->Fill(4);
    if (check_mjj700_Pt40_DEta3p5 == true)
      v.getMEhisto_NumberOfEvents()->Fill(5);
    if (check_mjj750_Pt40_DEta3p5 == true)
      v.getMEhisto_NumberOfEvents()->Fill(6);
    if (check_mjj800_Pt40_DEta3p5 == true)
      v.getMEhisto_NumberOfEvents()->Fill(7);
  }
}

// BeginRun
void HLTInclusiveVBFSource::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& c) {
  iBooker.setCurrentFolder(dirname_);

  //--- htlConfig_
  bool changed(true);
  if (!hltConfig_.init(run, c, processname_, changed)) {
    LogDebug("HLTInclusiveVBFSource") << "HLTConfigProvider failed to initialize.";
  }

  const unsigned int numberOfPaths(hltConfig_.size());
  for (unsigned int i = 0; i != numberOfPaths; ++i) {
    bool numFound = false;
    pathname = hltConfig_.triggerName(i);
    filtername = "dummy";
    unsigned int usedPrescale = 1;
    unsigned int objectType = 0;
    std::string triggerType = "";

    if (pathname.find("HLT_Di") == std::string::npos)
      continue;
    if (pathname.find("Jet") == std::string::npos)
      continue;
    if (pathname.find("MJJ") == std::string::npos)
      continue;
    if (pathname.find("VBF_v") == std::string::npos)
      continue;

    if (debug_) {
      cout << " - Startup:Path = " << pathname << endl;
      //cout<<" - Startup:PS = "<<hltConfig_.prescaleSize()<<endl;
    }

    triggerType = "DiJet_Trigger";
    objectType = trigger::TriggerJet;

    // Checking if the trigger exist in HLT table or not
    for (unsigned int i = 0; i != numberOfPaths; ++i) {
      std::string HLTname = hltConfig_.triggerName(i);
      if (HLTname == pathname)
        numFound = true;
    }

    if (numFound == false)
      continue;
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
    auto numpathmodule = numpathmodules.begin();
    for (; numpathmodule != numpathmodules.end(); ++numpathmodule) {
      edm::InputTag testTag(*numpathmodule, "", processname_);
      if (hltConfig_.moduleType(*numpathmodule) == "HLTCaloJetVBFFilter" ||
          hltConfig_.moduleType(*numpathmodule) == "HLTPFJetVBFFilter") {
        filtername = *numpathmodule;
        if (debug_)
          cout << " - Startup:Module = " << hltConfig_.moduleType(*numpathmodule) << ", FilterName = " << filtername
               << endl;
      }
    }
    if (debug_)
      cout << " - Startup:Final filter = " << filtername << endl;

    if (objectType == 0 || numFound == false)
      continue;
    //if(debug_){
    //cout<<"Pathname = "<<pathname
    //    <<", Filtername = "<<filtername
    //    <<", ObjectType = "<<objectType<<endl;
    //}
    hltPathsAll_.push_back(PathInfo(usedPrescale, pathname, filtername, processname_, objectType, triggerType));
  }  //Loop over paths

  //if(debug_) cout<<"== end hltPathsEff_.push_back ======" << endl;

  std::string dirName = dirname_ + "/MonitorInclusiveVBFTrigger/";
  for (auto& v : hltPathsAll_) {
    if (debug_)
      cout << "Storing: " << v.getPath() << ", Prescale = " << v.getprescaleUsed() << endl;
    //if(v->getprescaleUsed()!=1) continue;

    std::string subdirName = dirName + v.getPath();
    std::string trigPath = "(" + v.getPath() + ")";
    iBooker.setCurrentFolder(subdirName);

    MonitorElement* RECO_deltaEta_DiJet;
    MonitorElement* RECO_deltaPhi_DiJet;
    MonitorElement* RECO_invMass_DiJet;
    MonitorElement* HLT_deltaEta_DiJet;
    MonitorElement* HLT_deltaPhi_DiJet;
    MonitorElement* HLT_invMass_DiJet;
    MonitorElement* RECO_deltaEta_DiJet_Match;
    MonitorElement* RECO_deltaPhi_DiJet_Match;
    MonitorElement* RECO_invMass_DiJet_Match;
    MonitorElement* RECOHLT_deltaEta;
    MonitorElement* RECOHLT_deltaPhi;
    MonitorElement* RECOHLT_invMass;
    MonitorElement* NumberOfMatches;
    MonitorElement* NumberOfEvents;

    //dummy                     = iBooker.bookFloat("dummy");
    RECO_deltaEta_DiJet = iBooker.bookFloat("RECO_deltaEta_DiJet");
    RECO_deltaPhi_DiJet = iBooker.bookFloat("RECO_deltaPhi_DiJet");
    RECO_invMass_DiJet = iBooker.bookFloat("RECO_invMass_DiJet");
    HLT_deltaEta_DiJet = iBooker.bookFloat("HLT_deltaEta_DiJet");
    HLT_deltaPhi_DiJet = iBooker.bookFloat("HLT_deltaPhi_DiJet ");
    HLT_invMass_DiJet = iBooker.bookFloat("HLT_invMass_DiJet");
    RECO_deltaEta_DiJet_Match = iBooker.bookFloat("RECO_deltaEta_DiJet_Match");
    RECO_deltaPhi_DiJet_Match = iBooker.bookFloat("RECO_deltaPhi_DiJet_Match");
    RECO_invMass_DiJet_Match = iBooker.bookFloat("RECO_invMass_DiJet_Match");
    RECOHLT_deltaEta = iBooker.bookFloat("RECOHLT_deltaEta");
    RECOHLT_deltaPhi = iBooker.bookFloat("RECOHLT_deltaPhi ");
    RECOHLT_invMass = iBooker.bookFloat("RECOHLT_invMass");
    NumberOfMatches = iBooker.bookFloat("NumberOfMatches");
    NumberOfEvents = iBooker.bookFloat("NumberOfEvents");

    std::string labelname("ME");
    std::string histoname(labelname + "");
    std::string title(labelname + "");

    //RECO_deltaEta_DiJet
    histoname = labelname + "_RECO_deltaEta_DiJet";
    title = labelname + "_RECO_deltaEta_DiJet " + trigPath;
    RECO_deltaEta_DiJet = iBooker.book1D(histoname.c_str(), title.c_str(), 50, -10., 10.);
    RECO_deltaEta_DiJet->getTH1F();

    //RECO_deltaPhi_DiJet
    histoname = labelname + "_RECO_deltaPhi_DiJet";
    title = labelname + "_RECO_deltaPhi_DiJet " + trigPath;
    RECO_deltaPhi_DiJet = iBooker.book1D(histoname.c_str(), title.c_str(), 35, -3.5, 3.5);
    RECO_deltaPhi_DiJet->getTH1F();

    //RECO_invMass_DiJet
    histoname = labelname + "_RECO_invMass_DiJet";
    title = labelname + "_RECO_invMass_DiJet " + trigPath;
    RECO_invMass_DiJet = iBooker.book1D(histoname.c_str(), title.c_str(), 100, 500., 2000.);
    RECO_invMass_DiJet->getTH1F();

    //HLT_deltaEta_DiJet
    histoname = labelname + "_HLT_deltaEta_DiJet";
    title = labelname + "_HLT_deltaEta_DiJet " + trigPath;
    HLT_deltaEta_DiJet = iBooker.book1D(histoname.c_str(), title.c_str(), 50, -10., 10.);
    HLT_deltaEta_DiJet->getTH1F();

    //HLT_deltaPhi_DiJet
    histoname = labelname + "_HLT_deltaPhi_DiJet";
    title = labelname + "_HLT_deltaPhi_DiJet " + trigPath;
    HLT_deltaPhi_DiJet = iBooker.book1D(histoname.c_str(), title.c_str(), 35, -3.5, 3.5);
    HLT_deltaPhi_DiJet->getTH1F();

    //HLT_invMass_DiJet
    histoname = labelname + "_HLT_invMass_DiJet";
    title = labelname + "_HLT_invMass_DiJet " + trigPath;
    HLT_invMass_DiJet = iBooker.book1D(histoname.c_str(), title.c_str(), 100, 500., 2000.);
    HLT_invMass_DiJet->getTH1F();

    //RECO_deltaEta_DiJet_Match
    histoname = labelname + "_RECO_deltaEta_DiJet_Match";
    title = labelname + "_RECO_deltaEta_DiJet_Match " + trigPath;
    RECO_deltaEta_DiJet_Match = iBooker.book1D(histoname.c_str(), title.c_str(), 50, -10., 10.);
    RECO_deltaEta_DiJet_Match->getTH1F();

    //RECO_deltaPhi_DiJet_Match
    histoname = labelname + "_RECO_deltaPhi_DiJet_Match";
    title = labelname + "_RECO_deltaPhi_DiJet_Match " + trigPath;
    RECO_deltaPhi_DiJet_Match = iBooker.book1D(histoname.c_str(), title.c_str(), 35, -3.5, 3.5);
    RECO_deltaPhi_DiJet_Match->getTH1F();

    //RECO_invMass_DiJet_Match
    histoname = labelname + "_RECO_invMass_DiJet_Match";
    title = labelname + "_RECO_invMass_DiJet_Match " + trigPath;
    RECO_invMass_DiJet_Match = iBooker.book1D(histoname.c_str(), title.c_str(), 100, 500., 2000.);
    RECO_invMass_DiJet_Match->getTH1F();

    //RECOHLT_deltaEta
    histoname = labelname + "_RECOHLT_deltaEta";
    title = labelname + "_RECOHLT_deltaEta " + trigPath;
    RECOHLT_deltaEta = iBooker.book2D(histoname.c_str(), title.c_str(), 50, -10., 10., 50, -10., 10.);
    RECOHLT_deltaEta->getTH2F();

    //RECOHLT_deltaPhi
    histoname = labelname + "_RECOHLT_deltaPhi";
    title = labelname + "_RECOHLT_deltaPhi " + trigPath;
    RECOHLT_deltaPhi = iBooker.book2D(histoname.c_str(), title.c_str(), 35, -3.5, 3.5, 35, -3.5, 3.5);
    RECOHLT_deltaPhi->getTH2F();

    //RECOHLT_invMass
    histoname = labelname + "_RECOHLT_invMass";
    title = labelname + "_RECOHLT_invMass " + trigPath;
    RECOHLT_invMass = iBooker.book2D(histoname.c_str(), title.c_str(), 100, 500., 2000., 100, 500., 2000.);
    RECOHLT_invMass->getTH2F();

    //NumberOfMatches
    histoname = labelname + "_NumberOfMatches ";
    title = labelname + "_NumberOfMatches  " + trigPath;
    NumberOfMatches = iBooker.book1D(histoname.c_str(), title.c_str(), 2, 0., 2.);
    NumberOfMatches->getTH1F();

    //NumberOfEvents
    histoname = labelname + "_NumberOfEvents";
    title = labelname + "_NumberOfEvents " + trigPath;
    NumberOfEvents = iBooker.book1D(histoname.c_str(), title.c_str(), 10, 0., 10.);
    NumberOfEvents->getTH1F();

    //}
    v.setHistos(RECO_deltaEta_DiJet,
                RECO_deltaPhi_DiJet,
                RECO_invMass_DiJet,
                HLT_deltaEta_DiJet,
                HLT_deltaPhi_DiJet,
                HLT_invMass_DiJet,
                RECO_deltaEta_DiJet_Match,
                RECO_deltaPhi_DiJet_Match,
                RECO_invMass_DiJet_Match,
                RECOHLT_deltaEta,
                RECOHLT_deltaPhi,
                RECOHLT_invMass,
                NumberOfMatches,
                NumberOfEvents);
    //break;//We need only the first unprescale paths
  }
}

bool HLTInclusiveVBFSource::isBarrel(double eta) {
  bool output = false;
  if (fabs(eta) <= 1.3)
    output = true;
  return output;
}

bool HLTInclusiveVBFSource::isEndCap(double eta) {
  bool output = false;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3)
    output = true;
  return output;
}

bool HLTInclusiveVBFSource::isForward(double eta) {
  bool output = false;
  if (fabs(eta) > 3.0)
    output = true;
  return output;
}

bool HLTInclusiveVBFSource::validPathHLT(std::string pathname) {
  // hltConfig_ has to be defined first before calling this method
  bool output = false;
  for (unsigned int j = 0; j != hltConfig_.size(); ++j) {
    if (hltConfig_.triggerName(j) == pathname)
      output = true;
  }
  return output;
}

bool HLTInclusiveVBFSource::isHLTPathAccepted(std::string pathName) {
  // triggerResults_, triggerNames_ has to be defined first before calling this method
  bool output = false;
  if (triggerResults_.isValid()) {
    unsigned index = triggerNames_.triggerIndex(pathName);
    if (index < triggerNames_.size() && triggerResults_->accept(index))
      output = true;
  }
  return output;
}

bool HLTInclusiveVBFSource::isTriggerObjectFound(std::string objectName) {
  // processname_, triggerObj_ has to be defined before calling this method
  bool output = false;
  edm::InputTag testTag(objectName, "", processname_);
  const int index = triggerObj_->filterIndex(testTag);
  if (index >= triggerObj_->sizeFilters()) {
    edm::LogInfo("HLTInclusiveVBFSource") << "no index " << index << " of that name ";
  } else {
    const trigger::Keys& k = triggerObj_->filterKeys(index);
    if (!k.empty())
      output = true;
  }
  return output;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTInclusiveVBFSource);
