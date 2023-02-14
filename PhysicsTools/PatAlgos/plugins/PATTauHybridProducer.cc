#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/PATTauDiscriminator.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class PATTauHybridProducer : public edm::stream::EDProducer<> {
public:
  explicit PATTauHybridProducer(const edm::ParameterSet&);
  ~PATTauHybridProducer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  //--- configuration parameters
  edm::EDGetTokenT<pat::TauCollection> tausToken_;
  edm::EDGetTokenT<pat::JetCollection> jetsToken_;
  const float dRMax_, jetPtMin_, jetEtaMax_;
  const std::string pnetLabel_;
  std::vector<std::string> pnetTauScoreNames_;
  std::vector<std::string> pnetJetScoreNames_;
  std::vector<std::string> pnetLepScoreNames_;
  std::string pnetPtCorrName_;
  const float tauScoreMin_, chargeAssignmentProbMin_;
  const bool checkTauScoreIsBest_;

  const std::map<std::string, int> tagToDM_;
  enum class tauId_pnet_idx : size_t { dm, vsjet, vse, vsmu, ptcorr, last };
  enum class tauId_min_idx : size_t { hpsnew, last };
};

PATTauHybridProducer::PATTauHybridProducer(const edm::ParameterSet& cfg)
    : tausToken_(consumes<pat::TauCollection>(cfg.getParameter<edm::InputTag>("src"))),
      jetsToken_(consumes<pat::JetCollection>(cfg.getParameter<edm::InputTag>("jetSource"))),
      dRMax_(cfg.getParameter<double>("dRMax")),
      jetPtMin_(cfg.getParameter<double>("jetPtMin")),
      jetEtaMax_(cfg.getParameter<double>("jetEtaMax")),
      pnetLabel_(cfg.getParameter<std::string>("pnetLabel")),
      pnetPtCorrName_(cfg.getParameter<std::string>("pnetPtCorrName")),
      tauScoreMin_(cfg.getParameter<double>("tauScoreMin")),
      chargeAssignmentProbMin_(cfg.getParameter<double>("chargeAssignmentProbMin")),
      checkTauScoreIsBest_(cfg.getParameter<bool>("checkTauScoreIsBest")),
      tagToDM_({{"1h0p", 0}, {"1h1or2p", 1}, {"1h1p", 1}, {"1h2p", 2}, {"3h0p", 10}, {"3h1p", 11}}) {
  // read the different PNet score names
  std::vector<std::string> pnetScoreNames = cfg.getParameter<std::vector<std::string>>("pnetScoreNames");
  for (const auto& scoreName : pnetScoreNames) {
    size_t labelLenght = scoreName.find(':') == std::string::npos ? 0 : scoreName.find(':') + 1;
    std::string name = scoreName.substr(labelLenght);
    if (name.find("prob") == std::string::npos)
      continue;
    if (name.find("tau") != std::string::npos)
      pnetTauScoreNames_.push_back(name);
    else if (name.find("ele") != std::string::npos || name.find("mu") != std::string::npos)
      pnetLepScoreNames_.push_back(name);
    else
      pnetJetScoreNames_.push_back(name);
    if (pnetPtCorrName_.find(':') != std::string::npos)
      pnetPtCorrName_ = pnetPtCorrName_.substr(pnetPtCorrName_.find(':') + 1);
  }

  produces<std::vector<pat::Tau>>();
  //FIXME: produce a separate collection for PNet-recovered taus?
}

void PATTauHybridProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  // Get the vector of taus
  edm::Handle<pat::TauCollection> inputTaus;
  evt.getByToken(tausToken_, inputTaus);

  auto outputTaus = std::make_unique<std::vector<pat::Tau>>();
  outputTaus->reserve(inputTaus->size());

  // Get the vector of jets
  edm::Handle<pat::JetCollection> jets;
  evt.getByToken(jetsToken_, jets);

  //FIXME: How do the following using a global cache once per execution?
  //minimal HPS-like tauID list
  std::vector<pat::Tau::IdPair> tauIds_minimal((size_t)tauId_min_idx::last);
  tauIds_minimal[(size_t)tauId_min_idx::hpsnew] = std::make_pair("decayModeFindingNewDMs", -1);
  /* potentially can be added and computed when tau consistuents are defiend:
     'againstMuonLoose3' 'againstMuonTight3'
     'chargedIsoPtSum' 'chargedIsoPtSumdR03'
     'decayModeFinding' = -1 as newDMs by definition, add?
     'footprintCorrection' 'footprintCorrectiondR03' - usefull?
     'neutralIsoPtSum' 'neutralIsoPtSumdR03'
     'neutralIsoPtSumWeight' 'neutralIsoPtSumWeightdR03' - usefull?
     'photonPtSumOutsideSignalCone' 'photonPtSumOutsideSignalConedR03' - duplicated 
     'puCorrPtSum'*/

  // PNet tauID list
  std::vector<pat::Tau::IdPair> tauIds_pnet((size_t)tauId_pnet_idx::last);
  tauIds_pnet[(size_t)tauId_pnet_idx::dm] = std::make_pair("byPNetDecayMode", reco::PFTau::kNull);
  tauIds_pnet[(size_t)tauId_pnet_idx::vsjet] = std::make_pair("byPNetVSjetraw", -1);
  tauIds_pnet[(size_t)tauId_pnet_idx::vse] = std::make_pair("byPNetVSeraw", -1);
  tauIds_pnet[(size_t)tauId_pnet_idx::vsmu] = std::make_pair("byPNetVSmuraw", -1);
  tauIds_pnet[(size_t)tauId_pnet_idx::ptcorr] = std::make_pair("byPNetPtCorr", 1);

  std::set<unsigned int> matched_taus;
  size_t jet_idx = 0;
  for (const auto& jet : *jets) {
    jet_idx++;
    if (jet.pt() < jetPtMin_)
      continue;
    if (std::abs(jet.eta()) > jetEtaMax_)
      continue;
    size_t tau_idx = 0;
    bool matched = false;

    // Analyse PNet scores
    std::pair<std::string, float> bestPnetTauScore("probtauundef", -1);
    float sumOfPnetTauScores = 0;
    float plusChargeProb = 0;
    for (const auto& scoreName : pnetTauScoreNames_) {
      float score = jet.bDiscriminator(pnetLabel_ + ":" + scoreName);
      sumOfPnetTauScores += score;
      if (scoreName.find("taup") != std::string::npos)
        plusChargeProb += score;
      if (score > bestPnetTauScore.second) {
        bestPnetTauScore.second = score;
        bestPnetTauScore.first = scoreName;
      }
    }
    if (sumOfPnetTauScores > 0)
      plusChargeProb /= sumOfPnetTauScores;

    float sumOfPnetEleScores = 0, sumOfPnetMuScores = 0;
    bool isTauScoreBest = (sumOfPnetTauScores > 0);
    for (const auto& scoreName : pnetLepScoreNames_) {
      float score = jet.bDiscriminator(pnetLabel_ + ":" + scoreName);
      if (scoreName.find("ele") != std::string::npos)
        sumOfPnetEleScores += score;
      else if (scoreName.find("mu") != std::string::npos)
        sumOfPnetMuScores += score;
      if (score > bestPnetTauScore.second)
        isTauScoreBest = false;
    }
    if (checkTauScoreIsBest_ && isTauScoreBest) {  //if needed iterate over jet scores
      for (const auto& scoreName : pnetJetScoreNames_)
        if (jet.bDiscriminator(pnetLabel_ + ":" + scoreName) > bestPnetTauScore.second)
          isTauScoreBest = false;
    }

    // PNet discriminants vs jets, electrons and muons
    tauIds_pnet[(size_t)tauId_pnet_idx::vsjet].second =
        sumOfPnetTauScores /
        (1. - sumOfPnetEleScores -
         sumOfPnetMuScores);  //vsJet: tau scores by sum of tau and jet scores or equally by  1 - sum of lepton scores
    tauIds_pnet[(size_t)tauId_pnet_idx::vse].second =
        sumOfPnetTauScores /
        (sumOfPnetTauScores + sumOfPnetEleScores);  //vsEle: tau scores by sum of tau and ele scores
    tauIds_pnet[(size_t)tauId_pnet_idx::vsmu].second =
        sumOfPnetTauScores / (sumOfPnetTauScores + sumOfPnetMuScores);  //vsMu: tau scores by sum of tau and mu scores

    // Decay mode and charge of the highest tau score
    int bestCharge = 0;
    size_t pos =
        bestPnetTauScore.first.find("tau") + 3;  //this is well defined by constraction as name is "probtauXXXX"
    const char q = (pos < bestPnetTauScore.first.size()) ? bestPnetTauScore.first[pos] : 'u';
    if (q == 'm') {  //minus
      pos++;
      bestCharge = -1;
    } else if (q == 'p') {  //plus
      pos++;
      bestCharge = 1;
    }
    auto pNetDM = tagToDM_.find(bestPnetTauScore.first.substr(pos));
    if (pNetDM != tagToDM_.end())
      tauIds_pnet[(size_t)tauId_pnet_idx::dm].second = pNetDM->second;

    // PNet Pt correction
    float ptcorr = jet.bDiscriminator(pnetLabel_ + ":" + pnetPtCorrName_);
    if (ptcorr > -1000.)  // -1000. is default for not found discriminantor
      tauIds_pnet[(size_t)tauId_pnet_idx::ptcorr].second = ptcorr;

    // Search for matching tau
    for (const auto& inputTau : *inputTaus) {
      tau_idx++;
      if (matched_taus.count(tau_idx - 1) > 0)
        continue;
      float dR = deltaR(jet, inputTau);
      if (dR <
          dRMax_) {  //for now 1st fulfilling this requirement, one can look also for the closest tau with some overhead. But, for reasonable drMax_ the two should be equival in terms of found matches.
        matched_taus.insert(tau_idx - 1);
        pat::Tau outputTau(inputTau);
        const size_t nTauIds = inputTau.tauIDs().size();
        std::vector<pat::Tau::IdPair> tauIds(nTauIds + tauIds_pnet.size());
        for (size_t i = 0; i < nTauIds; ++i)
          tauIds[i] = inputTau.tauIDs()[i];
        for (size_t i = 0; i < tauIds_pnet.size(); ++i)
          tauIds[nTauIds + i] = tauIds_pnet[i];
        //FIXME: add jet ref, possible?
        outputTau.setTauIDs(tauIds);
        matched = true;
        outputTaus->push_back(outputTau);

        break;
      }
    }  // end of tau loop
    if (matched)
      continue;

    // Accept only jets passing minimal tau-like selection, i.e. with one of the tau score being globally the best and above some threshold, and with good quality of charge assignment
    if ((checkTauScoreIsBest_ && !isTauScoreBest) || bestPnetTauScore.second < tauScoreMin_ ||
        std::abs(0.5 - plusChargeProb) < chargeAssignmentProbMin_)
      continue;

    // Build taus from non-matched jets
    // "Null" pftau with jet kinematics
    reco::PFTau pfTauFromJet(bestCharge, jet.p4());
    //FIXME, find leading track (and best PV?), build patTau with DM, charge and discriminants by PNet, find possible consistunets for DM by PNet using HPS-like method
    //FIXME, the simplest way is to build cone-like PFtau cf. RecoTauTag/RecoTau/plugins/RecoTauBuilderConePlugin.cc

    // PATTau
    pat::Tau outputTauFromJet(pfTauFromJet);
    std::vector<pat::Tau::IdPair> newtauIds(tauIds_minimal.size() + tauIds_pnet.size());
    for (size_t i = 0; i < tauIds_minimal.size(); ++i)
      newtauIds[i] = tauIds_minimal[i];
    for (size_t i = 0; i < tauIds_pnet.size(); ++i)
      newtauIds[tauIds_minimal.size() + i] = tauIds_pnet[i];
    outputTauFromJet.setTauIDs(newtauIds);
    outputTaus->push_back(outputTauFromJet);

  }  // end of jet loop

  // Taus non-matched to jets (usually at pt-threshold or/and eta boundaries)
  if (matched_taus.size() < inputTaus->size()) {
    for (size_t i = 0; i < inputTaus->size(); ++i) {
      if (matched_taus.count(i) > 0)
        continue;
      const pat::Tau& inputTau = inputTaus->at(i);
      pat::Tau outputTau(inputTau);
      const size_t nTauIds = inputTau.tauIDs().size();
      std::vector<pat::Tau::IdPair> tauIds(nTauIds + tauIds_pnet.size());
      for (size_t i = 0; i < nTauIds; ++i)
        tauIds[i] = inputTau.tauIDs()[i];
      for (size_t i = 0; i < tauIds_pnet.size(); ++i) {
        tauIds[nTauIds + i] = tauIds_pnet[i];
        tauIds[nTauIds + i].second = -1;
      }
      outputTau.setTauIDs(tauIds);
      outputTaus->push_back(outputTau);
    }
  }  //non-matched taus

  evt.put(std::move(outputTaus));
}

void PATTauHybridProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // patTauHybridProducer
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("slimmedTaus"));
  desc.add<edm::InputTag>("jetSource", edm::InputTag("slimmedJetsUpadted"));
  desc.add<double>("dRMax", 0.4);
  desc.add<double>("jetPtMin", 20.0);
  desc.add<double>("jetEtaMax", 2.5);
  desc.add<std::string>("pnetLabel", "pfParticleNetAK4JetTags");
  desc.add<std::vector<std::string>>(
      "pnetScoreNames",
      {"probmu",       "probele",      "probtaup1h0p", "probtaup1h1p", "probtaup1h2p", "probtaup3h0p", "probtaup3h1p",
       "probtaum1h0p", "probtaum1h1p", "probtaum1h2p", "probtaum3h0p", "probtaum3h1p", "probb",        "probc",
       "probuds",      "probg",        "ptcorr",       "ptreshigh",    "ptreslow",     "ptnu"});
  desc.add<std::string>("pnetPtCorrName", "ptcorr");
  desc.add<double>("tauScoreMin", -1)->setComment("Minimal value of the best tau score to built recovery tau");
  desc.add<bool>("checkTauScoreIsBest", false)
      ->setComment("If true, recovery tau is built only if one of tau scores is the highest");
  desc.add<double>("chargeAssignmentProbMin", 0.2)
      ->setComment("Minimal value of charge assignment probability to built recovery tau (0,0.5)");

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTauHybridProducer);
