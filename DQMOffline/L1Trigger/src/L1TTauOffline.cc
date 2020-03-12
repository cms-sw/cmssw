/**
 *  @file     L1TTauOffline.cc
 *  @authors  Olivier Davignon (University of Bristol), Cécile Caillol (University of Wisconsin - Madison)
 *  @date     24/05/2017
 *  @version  1.1
 *
 */

#include "DQMOffline/L1Trigger/interface/L1TTauOffline.h"

#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonPFIsolation.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TLorentzVector.h"

#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace reco;
using namespace trigger;
using namespace edm;
using namespace std;

TauL1TPair::TauL1TPair(const TauL1TPair& tauL1tPair) {
  m_tau = tauL1tPair.m_tau;
  m_regTau = tauL1tPair.m_regTau;

  m_eta = tauL1tPair.m_eta;
  m_phi_bar = tauL1tPair.m_phi_bar;
  m_phi_end = tauL1tPair.m_phi_end;
}

double TauL1TPair::dR() { return deltaR(m_regTau->eta(), m_regTau->phi(), eta(), phi()); }
const std::map<std::string, unsigned int> L1TTauOffline::PlotConfigNames = {
    {"nVertex", PlotConfig::nVertex}, {"ETvsET", PlotConfig::ETvsET}, {"PHIvsPHI", PlotConfig::PHIvsPHI}};

//
// -------------------------------------- Constructor --------------------------------------------
//
L1TTauOffline::L1TTauOffline(const edm::ParameterSet& ps)
    : theTauCollection_(consumes<reco::PFTauCollection>(ps.getUntrackedParameter<edm::InputTag>("tauInputTag"))),
      AntiMuInputTag_(consumes<reco::PFTauDiscriminator>(ps.getUntrackedParameter<edm::InputTag>("antiMuInputTag"))),
      AntiEleInputTag_(consumes<reco::PFTauDiscriminator>(ps.getUntrackedParameter<edm::InputTag>("antiEleInputTag"))),
      DecayModeFindingInputTag_(
          consumes<reco::PFTauDiscriminator>(ps.getUntrackedParameter<edm::InputTag>("decayModeFindingInputTag"))),
      comb3TInputTag_(consumes<reco::PFTauDiscriminator>(ps.getUntrackedParameter<edm::InputTag>("comb3TInputTag"))),
      MuonInputTag_(consumes<reco::MuonCollection>(ps.getUntrackedParameter<edm::InputTag>("muonInputTag"))),
      MetInputTag_(consumes<reco::PFMETCollection>(ps.getUntrackedParameter<edm::InputTag>("metInputTag"))),
      VtxInputTag_(consumes<reco::VertexCollection>(ps.getUntrackedParameter<edm::InputTag>("vtxInputTag"))),
      BsInputTag_(consumes<reco::BeamSpot>(ps.getUntrackedParameter<edm::InputTag>("bsInputTag"))),
      triggerEvent_(consumes<trigger::TriggerEvent>(ps.getUntrackedParameter<edm::InputTag>("trigInputTag"))),
      trigProcess_(ps.getUntrackedParameter<string>("trigProcess")),
      triggerResults_(consumes<edm::TriggerResults>(ps.getUntrackedParameter<edm::InputTag>("trigProcess_token"))),
      triggerPath_(ps.getUntrackedParameter<vector<std::string>>("triggerNames")),
      histFolder_(ps.getParameter<std::string>("histFolder")),
      efficiencyFolder_(histFolder_ + "/efficiency_raw"),
      stage2CaloLayer2TauToken_(consumes<l1t::TauBxCollection>(ps.getUntrackedParameter<edm::InputTag>("l1tInputTag"))),
      tauEfficiencyThresholds_(ps.getParameter<std::vector<int>>("tauEfficiencyThresholds")),
      tauEfficiencyBins_(ps.getParameter<std::vector<double>>("tauEfficiencyBins")),
      histDefinitions_(dqmoffline::l1t::readHistDefinitions(ps.getParameterSet("histDefinitions"), PlotConfigNames)),
      m_TightMuons(),
      m_ProbeTaus(),
      m_TauL1tPairs(),
      m_RecoTaus(),
      m_L1tTaus(),
      m_RecoRecoTaus(),
      m_L1tL1tTaus(),
      m_L1tPtCuts(),
      m_MaxTauEta(99999),
      m_MaxL1tTauDR(99999),
      m_MaxHltTauDR(99999),
      m_trigIndices(),
      h_nVertex_(),
      h_tagAndProbeMass_(),
      h_L1TauETvsTauET_EB_(),
      h_L1TauETvsTauET_EE_(),
      h_L1TauETvsTauET_EB_EE_(),
      h_L1TauPhivsTauPhi_EB_(),
      h_L1TauPhivsTauPhi_EE_(),
      h_L1TauPhivsTauPhi_EB_EE_(),
      h_L1TauEtavsTauEta_(),
      h_resolutionTauET_EB_(),
      h_resolutionTauET_EE_(),
      h_resolutionTauET_EB_EE_(),
      h_resolutionTauPhi_EB_(),
      h_resolutionTauPhi_EE_(),
      h_resolutionTauPhi_EB_EE_(),
      h_resolutionTauEta_(),
      h_efficiencyIsoTauET_EB_pass_(),
      h_efficiencyIsoTauET_EE_pass_(),
      h_efficiencyIsoTauET_EB_EE_pass_(),
      h_efficiencyNonIsoTauET_EB_pass_(),
      h_efficiencyNonIsoTauET_EE_pass_(),
      h_efficiencyNonIsoTauET_EB_EE_pass_(),
      h_efficiencyIsoTauET_EB_total_(),
      h_efficiencyIsoTauET_EE_total_(),
      h_efficiencyIsoTauET_EB_EE_total_(),
      h_efficiencyNonIsoTauET_EB_total_(),
      h_efficiencyNonIsoTauET_EE_total_(),
      h_efficiencyNonIsoTauET_EB_EE_total_() {
  edm::LogInfo("L1TTauOffline") << "Constructor "
                                << "L1TTauOffline::L1TTauOffline " << std::endl;
}

//
// -- Destructor
//
L1TTauOffline::~L1TTauOffline() {
  edm::LogInfo("L1TTauOffline") << "Destructor L1TTauOffline::~L1TTauOffline " << std::endl;
}

//
// -------------------------------------- beginRun --------------------------------------------
//
void L1TTauOffline::dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup)
// void L1TTauOffline::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
  bool changed = true;
  m_hltConfig.init(run, iSetup, trigProcess_, changed);

  edm::LogInfo("L1TTauOffline") << "L1TTauOffline::beginRun" << std::endl;
}

//
// -------------------------------------- bookHistos --------------------------------------------
//
void L1TTauOffline::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  edm::LogInfo("L1TTauOffline") << "L1TTauOffline::bookHistograms" << std::endl;

  // book at beginRun
  bookTauHistos(ibooker);

  for (auto trigNamesIt = triggerPath_.begin(); trigNamesIt != triggerPath_.end(); trigNamesIt++) {
    std::string tNameTmp = (*trigNamesIt);
    std::string tNamePattern = "";
    std::size_t found0 = tNameTmp.find("*");
    if (found0 != std::string::npos)
      tNamePattern = tNameTmp.substr(0, tNameTmp.size() - 1);
    else
      tNamePattern = tNameTmp;

    int tIndex = -1;

    for (unsigned ipath = 0; ipath < m_hltConfig.size(); ++ipath) {
      std::string tmpName = m_hltConfig.triggerName(ipath);

      std::size_t found = tmpName.find(tNamePattern);
      if (found != std::string::npos) {
        tIndex = int(ipath);
        m_trigIndices.push_back(tIndex);
      }
    }
  }
}

//
// -------------------------------------- Analyze --------------------------------------------
//
void L1TTauOffline::analyze(edm::Event const& e, edm::EventSetup const& eSetup) {
  m_MaxTauEta = 2.1;
  m_MaxL1tTauDR = 0.5;
  m_MaxHltTauDR = 0.5;

  edm::Handle<reco::PFTauCollection> taus;
  e.getByToken(theTauCollection_, taus);

  if (!taus.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: reco::PFTauCollection " << std::endl;
    return;
  }

  edm::Handle<reco::MuonCollection> muons;
  e.getByToken(MuonInputTag_, muons);

  if (!muons.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: reco::MuonCollection " << std::endl;
    return;
  }

  edm::Handle<reco::BeamSpot> beamSpot;
  e.getByToken(BsInputTag_, beamSpot);

  if (!beamSpot.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: reco::BeamSpot " << std::endl;
    return;
  }

  edm::Handle<reco::VertexCollection> vertex;
  e.getByToken(VtxInputTag_, vertex);

  if (!vertex.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: reco::VertexCollection " << std::endl;
    return;
  }

  edm::Handle<l1t::TauBxCollection> l1tCands;
  e.getByToken(stage2CaloLayer2TauToken_, l1tCands);

  if (!l1tCands.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: l1t::TauBxCollection " << std::endl;
    return;
  }

  edm::Handle<edm::TriggerResults> trigResults;
  e.getByToken(triggerResults_, trigResults);

  if (!trigResults.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: edm::TriggerResults " << std::endl;
    return;
  }

  edm::Handle<trigger::TriggerEvent> trigEvent;
  e.getByToken(triggerEvent_, trigEvent);

  if (!trigEvent.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: trigger::TriggerEvent " << std::endl;
    return;
  }

  edm::Handle<reco::PFMETCollection> mets;
  e.getByToken(MetInputTag_, mets);

  if (!mets.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: reco::PFMETCollection " << std::endl;
    return;
  }

  eSetup.get<IdealMagneticFieldRecord>().get(m_BField);
  const reco::Vertex primaryVertex = getPrimaryVertex(vertex, beamSpot);

  getTightMuons(muons, mets, primaryVertex, trigEvent);
  getProbeTaus(e, taus, muons, primaryVertex);
  getTauL1tPairs(l1tCands);

  vector<l1t::Tau> l1tContainer;
  l1tContainer.reserve(l1tCands->size() + 1);

  for (auto tau = l1tCands->begin(0); tau != l1tCands->end(0); ++tau) {
    l1tContainer.push_back(*tau);
  }

  for (auto tauL1tPairsIt = m_TauL1tPairs.begin(); tauL1tPairsIt != m_TauL1tPairs.end(); ++tauL1tPairsIt) {
    float eta = tauL1tPairsIt->eta();
    float phi = tauL1tPairsIt->phi();
    float pt = tauL1tPairsIt->pt();

    // unmatched gmt cands have l1tPt = -1.
    float l1tPt = tauL1tPairsIt->l1tPt();

    int counter = 0;

    for (auto threshold : tauEfficiencyThresholds_) {
      std::string str_threshold = std::to_string(threshold);

      int l1tPtCut = threshold;
      bool l1tAboveCut = (l1tPt >= l1tPtCut);

      stringstream ptCutToTag;
      ptCutToTag << l1tPtCut;
      string ptTag = ptCutToTag.str();

      if (fabs(eta) < m_MaxTauEta) {
        if (counter == 0) {
          if (fabs(eta) < 1.5) {
            h_L1TauETvsTauET_EB_->Fill(pt, l1tPt);
            h_L1TauPhivsTauPhi_EB_->Fill(phi, tauL1tPairsIt->l1tPhi());
            h_resolutionTauET_EB_->Fill((l1tPt - pt) / pt);
            h_resolutionTauPhi_EB_->Fill(tauL1tPairsIt->l1tPhi() - phi);
          } else {
            h_L1TauETvsTauET_EE_->Fill(pt, l1tPt);
            h_L1TauPhivsTauPhi_EE_->Fill(phi, tauL1tPairsIt->l1tPhi());
            h_resolutionTauET_EE_->Fill((l1tPt - pt) / pt);
            h_resolutionTauPhi_EE_->Fill(tauL1tPairsIt->l1tPhi() - phi);
          }
          h_L1TauETvsTauET_EB_EE_->Fill(pt, l1tPt);
          h_L1TauPhivsTauPhi_EB_EE_->Fill(phi, tauL1tPairsIt->l1tPhi());
          h_L1TauEtavsTauEta_->Fill(eta, tauL1tPairsIt->l1tEta());
          h_resolutionTauET_EB_EE_->Fill((l1tPt - pt) / pt);
          h_resolutionTauPhi_EB_EE_->Fill(tauL1tPairsIt->l1tPhi() - phi);
          h_resolutionTauEta_->Fill(tauL1tPairsIt->l1tEta() - eta);

          ++counter;
        }

        if (fabs(eta) < 1.5) {
          h_efficiencyNonIsoTauET_EB_total_[threshold]->Fill(pt);
          h_efficiencyIsoTauET_EB_total_[threshold]->Fill(pt);
        } else {
          h_efficiencyNonIsoTauET_EE_total_[threshold]->Fill(pt);
          h_efficiencyIsoTauET_EE_total_[threshold]->Fill(pt);
        }
        h_efficiencyNonIsoTauET_EB_EE_total_[threshold]->Fill(pt);
        h_efficiencyIsoTauET_EB_EE_total_[threshold]->Fill(pt);

        if (l1tAboveCut) {
          if (fabs(eta) < 1.5)
            h_efficiencyNonIsoTauET_EB_pass_[threshold]->Fill(pt);
          else
            h_efficiencyNonIsoTauET_EE_pass_[threshold]->Fill(pt);
          h_efficiencyNonIsoTauET_EB_EE_pass_[threshold]->Fill(pt);

          if (tauL1tPairsIt->l1tIso() > 0.5) {
            if (fabs(eta) < 1.5)
              h_efficiencyIsoTauET_EB_pass_[threshold]->Fill(pt);
            else
              h_efficiencyIsoTauET_EE_pass_[threshold]->Fill(pt);
            h_efficiencyIsoTauET_EB_EE_pass_[threshold]->Fill(pt);
          }
        }
      }
    }
  }  // loop over tau-L1 pairs
}

//
// -------------------------------------- endRun --------------------------------------------
//
//
// -------------------------------------- book histograms --------------------------------------------
//
void L1TTauOffline::bookTauHistos(DQMStore::IBooker& ibooker) {
  ibooker.cd();
  ibooker.setCurrentFolder(histFolder_);
  dqmoffline::l1t::HistDefinition nVertexDef = histDefinitions_[PlotConfig::nVertex];
  h_nVertex_ = ibooker.book1D(nVertexDef.name, nVertexDef.title, nVertexDef.nbinsX, nVertexDef.xmin, nVertexDef.xmax);
  h_tagAndProbeMass_ = ibooker.book1D("tagAndProbeMass", "Invariant mass of tag & probe pair", 100, 40, 140);

  dqmoffline::l1t::HistDefinition templateETvsET = histDefinitions_[PlotConfig::ETvsET];
  h_L1TauETvsTauET_EB_ = ibooker.book2D("L1TauETvsTauET_EB",
                                        "L1 Tau E_{T} vs PFTau E_{T} (EB); PFTau E_{T} (GeV); L1 Tau E_{T} (GeV)",
                                        templateETvsET.nbinsX,
                                        &templateETvsET.binsX[0],
                                        templateETvsET.nbinsY,
                                        &templateETvsET.binsY[0]);
  h_L1TauETvsTauET_EE_ = ibooker.book2D("L1TauETvsTauET_EE",
                                        "L1 Tau E_{T} vs PFTau E_{T} (EE); PFTau E_{T} (GeV); L1 Tau E_{T} (GeV)",
                                        templateETvsET.nbinsX,
                                        &templateETvsET.binsX[0],
                                        templateETvsET.nbinsY,
                                        &templateETvsET.binsY[0]);
  h_L1TauETvsTauET_EB_EE_ = ibooker.book2D("L1TauETvsTauET_EB_EE",
                                           "L1 Tau E_{T} vs PFTau E_{T} (EB+EE); PFTau E_{T} (GeV); L1 Tau E_{T} (GeV)",
                                           templateETvsET.nbinsX,
                                           &templateETvsET.binsX[0],
                                           templateETvsET.nbinsY,
                                           &templateETvsET.binsY[0]);

  dqmoffline::l1t::HistDefinition templatePHIvsPHI = histDefinitions_[PlotConfig::PHIvsPHI];
  h_L1TauPhivsTauPhi_EB_ =
      ibooker.book2D("L1TauPhivsTauPhi_EB",
                     "#phi_{tau}^{L1} vs #phi_{tau}^{offline} (EB); #phi_{tau}^{offline}; #phi_{tau}^{L1}",
                     templatePHIvsPHI.nbinsX,
                     templatePHIvsPHI.xmin,
                     templatePHIvsPHI.xmax,
                     templatePHIvsPHI.nbinsY,
                     templatePHIvsPHI.ymin,
                     templatePHIvsPHI.ymax);
  h_L1TauPhivsTauPhi_EE_ =
      ibooker.book2D("L1TauPhivsTauPhi_EE",
                     "#phi_{tau}^{L1} vs #phi_{tau}^{offline} (EE); #phi_{tau}^{offline}; #phi_{tau}^{L1}",
                     templatePHIvsPHI.nbinsX,
                     templatePHIvsPHI.xmin,
                     templatePHIvsPHI.xmax,
                     templatePHIvsPHI.nbinsY,
                     templatePHIvsPHI.ymin,
                     templatePHIvsPHI.ymax);
  h_L1TauPhivsTauPhi_EB_EE_ =
      ibooker.book2D("L1TauPhivsTauPhi_EB_EE",
                     "#phi_{tau}^{L1} vs #phi_{tau}^{offline} (EB+EE); #phi_{tau}^{offline}; #phi_{tau}^{L1}",
                     templatePHIvsPHI.nbinsX,
                     templatePHIvsPHI.xmin,
                     templatePHIvsPHI.xmax,
                     templatePHIvsPHI.nbinsY,
                     templatePHIvsPHI.ymin,
                     templatePHIvsPHI.ymax);

  h_L1TauEtavsTauEta_ =
      ibooker.book2D("L1TauEtavsTauEta", "L1 Tau #eta vs PFTau #eta; PFTau #eta; L1 Tau #eta", 100, -3, 3, 100, -3, 3);

  // tau resolutions
  h_resolutionTauET_EB_ = ibooker.book1D(
      "resolutionTauET_EB", "tau ET resolution (EB); (L1 Tau E_{T} - PFTau E_{T})/PFTau E_{T}; events", 50, -1, 1.5);
  h_resolutionTauET_EE_ = ibooker.book1D(
      "resolutionTauET_EE", "tau ET resolution (EE); (L1 Tau E_{T} - PFTau E_{T})/PFTau E_{T}; events", 50, -1, 1.5);
  h_resolutionTauET_EB_EE_ =
      ibooker.book1D("resolutionTauET_EB_EE",
                     "tau ET resolution (EB+EE); (L1 Tau E_{T} - PFTau E_{T})/PFTau E_{T}; events",
                     50,
                     -1,
                     1.5);

  h_resolutionTauPhi_EB_ = ibooker.book1D("resolutionTauPhi_EB",
                                          "#phi_{tau} resolution (EB); #phi_{tau}^{L1} - #phi_{tau}^{offline}; events",
                                          120,
                                          -0.3,
                                          0.3);
  h_resolutionTauPhi_EE_ = ibooker.book1D(
      "resolutionTauPhi_EE", "tau #phi resolution (EE); #phi_{tau}^{L1} - #phi_{tau}^{offline}; events", 120, -0.3, 0.3);
  h_resolutionTauPhi_EB_EE_ =
      ibooker.book1D("resolutionTauPhi_EB_EE",
                     "tau #phi resolution (EB+EE); #phi_{tau}^{L1} - #phi_{tau}^{offline}; events",
                     120,
                     -0.3,
                     0.3);

  h_resolutionTauEta_ =
      ibooker.book1D("resolutionTauEta", "tau #eta resolution  (EB); L1 Tau #eta - PFTau #eta; events", 120, -0.3, 0.3);

  // tau turn-ons
  ibooker.setCurrentFolder(efficiencyFolder_);
  std::vector<float> tauBins(tauEfficiencyBins_.begin(), tauEfficiencyBins_.end());
  int nBins = tauBins.size() - 1;
  float* tauBinArray = &(tauBins[0]);

  for (auto threshold : tauEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyIsoTauET_EB_pass_[threshold] =
        ibooker.book1D("efficiencyIsoTauET_EB_threshold_" + str_threshold + "_Num",
                       "iso tau efficiency (EB); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);
    h_efficiencyIsoTauET_EE_pass_[threshold] =
        ibooker.book1D("efficiencyIsoTauET_EE_threshold_" + str_threshold + "_Num",
                       "iso tau efficiency (EE); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);
    h_efficiencyIsoTauET_EB_EE_pass_[threshold] =
        ibooker.book1D("efficiencyIsoTauET_EB_EE_threshold_" + str_threshold + "_Num",
                       "iso tau efficiency (EB+EE); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);

    h_efficiencyIsoTauET_EB_total_[threshold] =
        ibooker.book1D("efficiencyIsoTauET_EB_threshold_" + str_threshold + "_Den",
                       "iso tau efficiency (EB); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);
    h_efficiencyIsoTauET_EE_total_[threshold] =
        ibooker.book1D("efficiencyIsoTauET_EE_threshold_" + str_threshold + "_Den",
                       "iso tau efficiency (EE); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);
    h_efficiencyIsoTauET_EB_EE_total_[threshold] =
        ibooker.book1D("efficiencyIsoTauET_EB_EE_threshold_" + str_threshold + "_Den",
                       "iso tau efficiency (EB+EE); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);

    // non iso
    h_efficiencyNonIsoTauET_EB_pass_[threshold] =
        ibooker.book1D("efficiencyNonIsoTauET_EB_threshold_" + str_threshold + "_Num",
                       "inclusive tau efficiency (EB); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);
    h_efficiencyNonIsoTauET_EE_pass_[threshold] =
        ibooker.book1D("efficiencyNonIsoTauET_EE_threshold_" + str_threshold + "_Num",
                       "inclusive tau efficiency (EE); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);
    h_efficiencyNonIsoTauET_EB_EE_pass_[threshold] =
        ibooker.book1D("efficiencyNonIsoTauET_EB_EE_threshold_" + str_threshold + "_Num",
                       "inclusive tau efficiency (EB+EE); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);

    h_efficiencyNonIsoTauET_EB_total_[threshold] =
        ibooker.book1D("efficiencyNonIsoTauET_EB_threshold_" + str_threshold + "_Den",
                       "inclusive tau efficiency (EB); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);
    h_efficiencyNonIsoTauET_EE_total_[threshold] =
        ibooker.book1D("efficiencyNonIsoTauET_EE_threshold_" + str_threshold + "_Den",
                       "inclusive tau efficiency (EE); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);
    h_efficiencyNonIsoTauET_EB_EE_total_[threshold] =
        ibooker.book1D("efficiencyNonIsoTauET_EB_EE_threshold_" + str_threshold + "_Den",
                       "inclusive tau efficiency (EB+EE); PFTau E_{T} (GeV); events",
                       nBins,
                       tauBinArray);
  }

  ibooker.cd();

  return;
}

const reco::Vertex L1TTauOffline::getPrimaryVertex(edm::Handle<reco::VertexCollection> const& vertex,
                                                   edm::Handle<reco::BeamSpot> const& beamSpot) {
  reco::Vertex::Point posVtx;
  reco::Vertex::Error errVtx;

  bool hasPrimaryVertex = false;

  if (vertex.isValid()) {
    for (auto vertexIt = vertex->begin(); vertexIt != vertex->end(); ++vertexIt) {
      if (vertexIt->isValid() && !vertexIt->isFake()) {
        posVtx = vertexIt->position();
        errVtx = vertexIt->error();
        hasPrimaryVertex = true;
        break;
      }
    }
  }

  if (!hasPrimaryVertex) {
    posVtx = beamSpot->position();
    errVtx(0, 0) = beamSpot->BeamWidthX();
    errVtx(1, 1) = beamSpot->BeamWidthY();
    errVtx(2, 2) = beamSpot->sigmaZ();
  }

  const reco::Vertex primaryVertex(posVtx, errVtx);

  return primaryVertex;
}

bool L1TTauOffline::matchHlt(edm::Handle<trigger::TriggerEvent> const& triggerEvent, const reco::Muon* muon) {
  double matchDeltaR = 9999;

  trigger::TriggerObjectCollection trigObjs = triggerEvent->getObjects();

  for (auto trigIndexIt = m_trigIndices.begin(); trigIndexIt != m_trigIndices.end(); ++trigIndexIt) {
    const vector<string> moduleLabels(m_hltConfig.moduleLabels(*trigIndexIt));
    const unsigned moduleIndex = m_hltConfig.size((*trigIndexIt)) - 2;

    const unsigned hltFilterIndex = triggerEvent->filterIndex(InputTag(moduleLabels[moduleIndex], "", trigProcess_));

    if (hltFilterIndex < triggerEvent->sizeFilters()) {
      const Keys triggerKeys(triggerEvent->filterKeys(hltFilterIndex));
      const Vids triggerVids(triggerEvent->filterIds(hltFilterIndex));

      const unsigned nTriggers = triggerVids.size();
      for (size_t iTrig = 0; iTrig < nTriggers; ++iTrig) {
        const TriggerObject trigObject = trigObjs[triggerKeys[iTrig]];

        double dRtmp = deltaR((*muon), trigObject);
        if (dRtmp < matchDeltaR)
          matchDeltaR = dRtmp;
      }
    }
  }

  return (matchDeltaR < m_MaxHltTauDR);
}

void L1TTauOffline::getTauL1tPairs(edm::Handle<l1t::TauBxCollection> const& l1tCands) {
  m_TauL1tPairs.clear();

  vector<l1t::Tau> l1tContainer;
  l1tContainer.reserve(l1tCands->size() + 1);

  for (auto tau = l1tCands->begin(0); tau != l1tCands->end(0); ++tau) {
    l1tContainer.push_back(*tau);
  }

  for (auto probeTauIt = m_ProbeTaus.begin(); probeTauIt != m_ProbeTaus.end(); ++probeTauIt) {
    TauL1TPair pairBestCand((*probeTauIt), nullptr);

    for (auto l1tIt = l1tContainer.begin(); l1tIt != l1tContainer.end(); ++l1tIt) {
      TauL1TPair pairTmpCand((*probeTauIt), &(*l1tIt));

      if (pairTmpCand.dR() < m_MaxL1tTauDR && pairTmpCand.l1tPt() > pairBestCand.l1tPt())
        pairBestCand = pairTmpCand;
    }

    m_TauL1tPairs.push_back(pairBestCand);
  }
}

void L1TTauOffline::getTightMuons(edm::Handle<reco::MuonCollection> const& muons,
                                  edm::Handle<reco::PFMETCollection> const& mets,
                                  const reco::Vertex& vertex,
                                  edm::Handle<trigger::TriggerEvent> const& trigEvent) {
  m_TightMuons.clear();

  const reco::PFMET* pfmet = nullptr;
  pfmet = &(mets->front());

  int nb_mu = 0;

  for (auto muonIt2 = muons->begin(); muonIt2 != muons->end(); ++muonIt2) {
    if (fabs(muonIt2->eta()) < 2.4 && muonIt2->pt() > 10 && muon::isLooseMuon((*muonIt2)) &&
        (muonIt2->pfIsolationR04().sumChargedHadronPt +
         max(muonIt2->pfIsolationR04().sumNeutralHadronEt + muonIt2->pfIsolationR04().sumPhotonEt -
                 0.5 * muonIt2->pfIsolationR04().sumPUPt,
             0.0)) /
                muonIt2->pt() <
            0.3) {
      ++nb_mu;
    }
  }
  bool foundTightMu = false;
  for (auto muonIt = muons->begin(); muonIt != muons->end(); ++muonIt) {
    if (!matchHlt(trigEvent, &(*muonIt)))
      continue;
    float muiso = (muonIt->pfIsolationR04().sumChargedHadronPt +
                   max(muonIt->pfIsolationR04().sumNeutralHadronEt + muonIt->pfIsolationR04().sumPhotonEt -
                           0.5 * muonIt->pfIsolationR04().sumPUPt,
                       0.0)) /
                  muonIt->pt();

    if (muiso < 0.1 && nb_mu < 2 && !foundTightMu && fabs(muonIt->eta()) < 2.1 && muonIt->pt() > 24 &&
        muon::isLooseMuon((*muonIt))) {
      float mt = sqrt(pow(muonIt->pt() + pfmet->pt(), 2) - pow(muonIt->px() + pfmet->px(), 2) -
                      pow(muonIt->py() + pfmet->py(), 2));
      if (mt < 30) {
        m_TightMuons.push_back(&(*muonIt));
        foundTightMu = true;
      }
    }
  }
}

void L1TTauOffline::getProbeTaus(const edm::Event& iEvent,
                                 edm::Handle<reco::PFTauCollection> const& taus,
                                 edm::Handle<reco::MuonCollection> const& muons,
                                 const reco::Vertex& vertex) {
  m_ProbeTaus.clear();

  edm::Handle<reco::PFTauDiscriminator> antimu;
  iEvent.getByToken(AntiMuInputTag_, antimu);
  if (!antimu.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: reco::PFTauDiscriminator " << std::endl;
    return;
  }

  edm::Handle<reco::PFTauDiscriminator> dmf;
  iEvent.getByToken(DecayModeFindingInputTag_, dmf);
  if (!dmf.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: reco::PFTauDiscriminator " << std::endl;
    return;
  }

  edm::Handle<reco::PFTauDiscriminator> antiele;
  iEvent.getByToken(AntiEleInputTag_, antiele);
  if (!antiele.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: reco::PFTauDiscriminator " << std::endl;
    return;
  }

  edm::Handle<reco::PFTauDiscriminator> comb3T;
  iEvent.getByToken(comb3TInputTag_, comb3T);
  if (!comb3T.isValid()) {
    edm::LogWarning("L1TTauOffline") << "invalid collection: reco::PFTauDiscriminator " << std::endl;
    return;
  }

  if (!m_TightMuons.empty()) {
    TLorentzVector mymu;
    mymu.SetPtEtaPhiE(m_TightMuons[0]->pt(), m_TightMuons[0]->eta(), m_TightMuons[0]->phi(), m_TightMuons[0]->energy());
    int iTau = 0;
    for (auto tauIt = taus->begin(); tauIt != taus->end(); ++tauIt, ++iTau) {
      reco::PFTauRef tauCandidate(taus, iTau);
      TLorentzVector mytau;
      mytau.SetPtEtaPhiE(tauIt->pt(), tauIt->eta(), tauIt->phi(), tauIt->energy());

      if (fabs(tauIt->charge()) == 1 && fabs(tauIt->eta()) < 2.1 && tauIt->pt() > 20 && (*antimu)[tauCandidate] > 0.5 &&
          (*antiele)[tauCandidate] > 0.5 && (*dmf)[tauCandidate] > 0.5 && (*comb3T)[tauCandidate] > 0.5) {
        if (mymu.DeltaR(mytau) > 0.5 && (mymu + mytau).M() > 40 && (mymu + mytau).M() < 80 &&
            m_TightMuons[0]->charge() * tauIt->charge() < 0) {
          m_ProbeTaus.push_back(&(*tauIt));
        }
      }
    }
  }
}

void L1TTauOffline::normalise2DHistogramsToBinArea() {
  std::vector<MonitorElement*> monElementstoNormalize = {h_L1TauETvsTauET_EB_,
                                                         h_L1TauETvsTauET_EE_,
                                                         h_L1TauETvsTauET_EB_EE_,
                                                         h_L1TauPhivsTauPhi_EB_,
                                                         h_L1TauPhivsTauPhi_EE_,
                                                         h_L1TauPhivsTauPhi_EB_EE_,
                                                         h_L1TauEtavsTauEta_};

  for (auto mon : monElementstoNormalize) {
    if (mon != nullptr) {
      auto h = mon->getTH2F();
      if (h != nullptr) {
        h->Scale(1, "width");
      }
    }
  }
}
// define this as a plug-in
DEFINE_FWK_MODULE(L1TTauOffline);
