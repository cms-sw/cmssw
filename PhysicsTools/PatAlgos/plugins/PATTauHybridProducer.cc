#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "CommonTools/Utils/interface/PtComparator.h"

class PATTauHybridProducer : public edm::stream::EDProducer<> {
public:
  explicit PATTauHybridProducer(const edm::ParameterSet&);
  ~PATTauHybridProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  void fillTauFromJet(reco::PFTau& pfTau, const reco::JetBaseRef& jet);
  //--- configuration parameters
  edm::EDGetTokenT<pat::TauCollection> tausToken_;
  edm::EDGetTokenT<pat::JetCollection> jetsToken_;
  bool addGenJetMatch_;
  edm::EDGetTokenT<edm::Association<reco::GenJetCollection>> genJetMatchToken_;
  const float dR2Max_, jetPtMin_, jetEtaMax_;
  const std::string UTagLabel_;
  const std::string tagPrefix_;
  std::vector<std::string> utagTauScoreNames_;
  std::vector<std::string> utagJetScoreNames_;
  std::vector<std::string> utagLepScoreNames_;
  std::string UtagPtCorrName_;
  const float tauScoreMin_, vsJetMin_, chargeAssignmentProbMin_;
  const bool checkTauScoreIsBest_;
  const bool usePFLeptonsAsChargedHadrons_;

  GreaterByPt<pat::Tau> pTTauComparator_;
  const std::map<std::string, int> tagToDM_;
  enum class tauId_utag_idx : size_t { dm = 0, vsjet, vse, vsmu, ptcorr, qconf, pdm0, pdm1, pdm2, pdm10, pdm11, last };
  enum class tauId_min_idx : size_t { hpsnew = 0, last };
};
PATTauHybridProducer::PATTauHybridProducer(const edm::ParameterSet& cfg)
    : tausToken_(consumes<pat::TauCollection>(cfg.getParameter<edm::InputTag>("src"))),
      jetsToken_(consumes<pat::JetCollection>(cfg.getParameter<edm::InputTag>("jetSource"))),
      addGenJetMatch_(cfg.getParameter<bool>("addGenJetMatch")),
      dR2Max_(std::pow(cfg.getParameter<double>("dRMax"), 2)),
      jetPtMin_(cfg.getParameter<double>("jetPtMin")),
      jetEtaMax_(cfg.getParameter<double>("jetEtaMax")),
      UTagLabel_(cfg.getParameter<std::string>("UTagLabel")),
      tagPrefix_(cfg.getParameter<std::string>("tagPrefix")),
      UtagPtCorrName_(cfg.getParameter<std::string>("UtagPtCorrName")),
      tauScoreMin_(cfg.getParameter<double>("tauScoreMin")),
      vsJetMin_(cfg.getParameter<double>("vsJetMin")),
      chargeAssignmentProbMin_(cfg.getParameter<double>("chargeAssignmentProbMin")),
      checkTauScoreIsBest_(cfg.getParameter<bool>("checkTauScoreIsBest")),
      usePFLeptonsAsChargedHadrons_(cfg.getParameter<bool>("usePFLeptonsAsChargedHadrons")),
      tagToDM_({{"1h0p", 0}, {"1h1or2p", 1}, {"1h1p", 1}, {"1h2p", 2}, {"3h0p", 10}, {"3h1p", 11}}) {
  // Read the different Unified Tagger score names
  std::vector<std::string> UTagScoreNames = cfg.getParameter<std::vector<std::string>>("UTagScoreNames");
  for (const auto& scoreName : UTagScoreNames) {
    // Check that discriminator matches tagger specified
    if (scoreName.find(UTagLabel_) == std::string::npos)
      continue;
    size_t labelLength = scoreName.find(':') == std::string::npos ? 0 : scoreName.find(':') + 1;
    std::string name = scoreName.substr(labelLength);
    if (name.find("prob") == std::string::npos)
      continue;
    if (name.find("probtau") != std::string::npos)
      utagTauScoreNames_.push_back(name);
    else if (name == "probele" || name == "probmu")
      utagLepScoreNames_.push_back(name);
    else if (name.find("data") == std::string::npos && name.find("mc") == std::string::npos)
      utagJetScoreNames_.push_back(name);
    if (UtagPtCorrName_.find(':') != std::string::npos)
      UtagPtCorrName_ = UtagPtCorrName_.substr(UtagPtCorrName_.find(':') + 1);
  }
  // GenJet matching
  if (addGenJetMatch_) {
    genJetMatchToken_ =
        consumes<edm::Association<reco::GenJetCollection>>(cfg.getParameter<edm::InputTag>("genJetMatch"));
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

  // Switch off gen-matching for real data
  if (evt.isRealData()) {
    addGenJetMatch_ = false;
  }
  edm::Handle<edm::Association<reco::GenJetCollection>> genJetMatch;
  if (addGenJetMatch_)
    evt.getByToken(genJetMatchToken_, genJetMatch);

  // Minimal HPS-like tauID list
  std::vector<pat::Tau::IdPair> tauIds_minimal((size_t)tauId_min_idx::last);
  tauIds_minimal[(size_t)tauId_min_idx::hpsnew] = std::make_pair("decayModeFindingNewDMs", -1);

  // Unified Tagger tauID list
  std::vector<pat::Tau::IdPair> tauIds_utag((size_t)tauId_utag_idx::last);
  tauIds_utag[(size_t)tauId_utag_idx::dm] = std::make_pair(tagPrefix_ + "DecayMode", reco::PFTau::kNull);
  tauIds_utag[(size_t)tauId_utag_idx::vsjet] = std::make_pair(tagPrefix_ + "VSjetraw", -1);
  tauIds_utag[(size_t)tauId_utag_idx::vse] = std::make_pair(tagPrefix_ + "VSeraw", -1);
  tauIds_utag[(size_t)tauId_utag_idx::vsmu] = std::make_pair(tagPrefix_ + "VSmuraw", -1);
  tauIds_utag[(size_t)tauId_utag_idx::ptcorr] = std::make_pair(tagPrefix_ + "PtCorr", 1);
  tauIds_utag[(size_t)tauId_utag_idx::qconf] = std::make_pair(tagPrefix_ + "QConf", 0);
  tauIds_utag[(size_t)tauId_utag_idx::pdm0] = std::make_pair(tagPrefix_ + "Prob1h0pi0", -1);
  tauIds_utag[(size_t)tauId_utag_idx::pdm1] = std::make_pair(tagPrefix_ + "Prob1h1pi0", -1);
  tauIds_utag[(size_t)tauId_utag_idx::pdm2] = std::make_pair(tagPrefix_ + "Prob1h2pi0", -1);
  tauIds_utag[(size_t)tauId_utag_idx::pdm10] = std::make_pair(tagPrefix_ + "Prob3h0pi0", -1);
  tauIds_utag[(size_t)tauId_utag_idx::pdm11] = std::make_pair(tagPrefix_ + "Prob3h1pi0", -1);

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

    // Analyse Tagger scores
    std::pair<std::string, float> bestUtagTauScore("probtauundef", -1);
    float sumOfUtagTauScores = 0;
    std::vector<float> tauPerDMScores(5);
    float plusChargeProb = 0;
    for (const auto& scoreName : utagTauScoreNames_) {
      float score = jet.bDiscriminator(UTagLabel_ + ":" + scoreName);
      sumOfUtagTauScores += score;
      if (scoreName.find("taup") != std::string::npos)
        plusChargeProb += score;
      if (score > bestUtagTauScore.second) {
        bestUtagTauScore.second = score;
        bestUtagTauScore.first = scoreName;
      }
      if (scoreName.find("1h0p") != std::string::npos)
        tauPerDMScores[0] += score;
      else if (scoreName.find("1h1") !=
               std::string::
                   npos)  //Note: final "p" in "1p" ommited to enble matching also with "1h1or2p" from early trainings
        tauPerDMScores[1] += score;
      else if (scoreName.find("1h2p") != std::string::npos)
        tauPerDMScores[2] += score;
      else if (scoreName.find("3h0p") != std::string::npos)
        tauPerDMScores[3] += score;
      else if (scoreName.find("3h1p") != std::string::npos)
        tauPerDMScores[4] += score;
    }
    if (sumOfUtagTauScores > 0)
      plusChargeProb /= sumOfUtagTauScores;

    float sumOfUtagEleScores = 0, sumOfUtagMuScores = 0;
    bool isTauScoreBest = (sumOfUtagTauScores > 0);
    for (const auto& scoreName : utagLepScoreNames_) {
      float score = jet.bDiscriminator(UTagLabel_ + ":" + scoreName);
      if (scoreName.find("ele") != std::string::npos)
        sumOfUtagEleScores += score;
      else if (scoreName.find("mu") != std::string::npos)
        sumOfUtagMuScores += score;
      if (score > bestUtagTauScore.second)
        isTauScoreBest = false;
    }
    if (checkTauScoreIsBest_ && isTauScoreBest) {  //if needed iterate over jet scores
      for (const auto& scoreName : utagJetScoreNames_)
        if (jet.bDiscriminator(UTagLabel_ + ":" + scoreName) > bestUtagTauScore.second)
          isTauScoreBest = false;
    }

    // Unified Tagger discriminants vs jets, electrons and muons
    tauIds_utag[(size_t)tauId_utag_idx::vsjet].second =
        sumOfUtagTauScores /
        (1. - sumOfUtagEleScores -
         sumOfUtagMuScores);  //vsJet: tau scores by sum of tau and jet scores or equally by  1 - sum of lepton scores
    tauIds_utag[(size_t)tauId_utag_idx::vse].second =
        sumOfUtagTauScores /
        (sumOfUtagTauScores + sumOfUtagEleScores);  //vsEle: tau scores by sum of tau and ele scores
    tauIds_utag[(size_t)tauId_utag_idx::vsmu].second =
        sumOfUtagTauScores / (sumOfUtagTauScores + sumOfUtagMuScores);  //vsMu: tau scores by sum of tau and mu scores

    // Decay mode and charge of the highest tau score
    int bestCharge = 0;
    size_t pos =
        bestUtagTauScore.first.find("tau") + 3;  //this is well defined by constraction as name is "probtauXXXX"
    const char q = (pos < bestUtagTauScore.first.size()) ? bestUtagTauScore.first[pos] : 'u';
    if (q == 'm') {  //minus
      pos++;
      bestCharge = -1;
    } else if (q == 'p') {  //plus
      pos++;
      bestCharge = 1;
    }
    auto UtagDM = tagToDM_.find(bestUtagTauScore.first.substr(pos));
    if (UtagDM != tagToDM_.end())
      tauIds_utag[(size_t)tauId_utag_idx::dm].second = UtagDM->second;

    // Unified tagger Pt correction
    float ptcorr = jet.bDiscriminator(UTagLabel_ + ":" + UtagPtCorrName_);
    if (ptcorr > -1000.)  // -1000. is default for not found discriminantor
      tauIds_utag[(size_t)tauId_utag_idx::ptcorr].second = ptcorr;

    // Unified Tagger charge confidence
    tauIds_utag[(size_t)tauId_utag_idx::qconf].second = (plusChargeProb - 0.5);

    // Unified Tagger per decay mode normalised score
    tauIds_utag[(size_t)tauId_utag_idx::pdm0].second = tauPerDMScores[0] / sumOfUtagTauScores;
    tauIds_utag[(size_t)tauId_utag_idx::pdm1].second = tauPerDMScores[1] / sumOfUtagTauScores;
    tauIds_utag[(size_t)tauId_utag_idx::pdm2].second = tauPerDMScores[2] / sumOfUtagTauScores;
    tauIds_utag[(size_t)tauId_utag_idx::pdm10].second = tauPerDMScores[3] / sumOfUtagTauScores;
    tauIds_utag[(size_t)tauId_utag_idx::pdm11].second = tauPerDMScores[4] / sumOfUtagTauScores;

    // Search for matching tau
    for (const auto& inputTau : *inputTaus) {
      tau_idx++;
      if (matched_taus.count(tau_idx - 1) > 0)
        continue;
      float dR2 = deltaR2(jet, inputTau);
      // select 1st found match rather than best match (both should be equivalent for reasonable dRMax)
      if (dR2 < dR2Max_) {
        matched_taus.insert(tau_idx - 1);
        pat::Tau outputTau(inputTau);
        const size_t nTauIds = inputTau.tauIDs().size();
        std::vector<pat::Tau::IdPair> tauIds(nTauIds + tauIds_utag.size());
        for (size_t i = 0; i < nTauIds; ++i)
          tauIds[i] = inputTau.tauIDs()[i];
        for (size_t i = 0; i < tauIds_utag.size(); ++i) {
          if ((tauIds_utag[i].first.find("PtCorr") != std::string::npos) &&
              (inputTau.tauID("decayModeFindingNewDMs") == -1)) {
            // if jet is matched to a recovered tau (i.e. non-HPS) then the Pt Correction
            // should be adjusted so that it can still be applied as PtCorr * TauPt
            // (as the original PtCorr will be w.r.t the jet pT, but recovered tau
            // pT is not necessarily set by same jet algorithm if adding both CHS
            // and PUPPI based taggers)
            tauIds[nTauIds + i].first = tauIds_utag[i].first;
            tauIds[nTauIds + i].second = tauIds_utag[i].second * jet.correctedP4("Uncorrected").pt() / inputTau.pt();
          } else {
            tauIds[nTauIds + i] = tauIds_utag[i];
          }
        }
        outputTau.setTauIDs(tauIds);
        matched = true;
        outputTaus->push_back(outputTau);

        break;
      }
    }  // end of tau loop
    if (matched)
      continue;

    // Accept only jets passing minimal tau-like selection, i.e. with one of the tau score being globally the best and above some threshold, and with good quality of charge assignment
    if ((checkTauScoreIsBest_ && !isTauScoreBest) || bestUtagTauScore.second < tauScoreMin_ ||
        tauIds_utag[(size_t)tauId_utag_idx::vsjet].second < vsJetMin_ ||
        std::abs(0.5 - plusChargeProb) < chargeAssignmentProbMin_)
      continue;

    // Build taus from non-matched jets
    // "Null" pftau with raw (uncorrected) jet kinematics
    reco::PFTau pfTauFromJet(bestCharge, jet.correctedP4("Uncorrected"));
    // Set PDGid
    pfTauFromJet.setPdgId(bestCharge < 0 ? 15 : -15);
    // and decay mode predicted by unified Tagger
    pfTauFromJet.setDecayMode(
        static_cast<const reco::PFTau::hadronicDecayMode>(int(tauIds_utag[(size_t)tauId_utag_idx::dm].second)));
    // Fill tau content using only jet consistunets within cone around leading
    // charged particle
    // FIXME: more sophisticated finding of tau constituents will be considered later
    pfTauFromJet.setSignalConeSize(
        std::clamp(3.6 / jet.correctedP4("Uncorrected").pt(), 0.08, 0.12));  // shrinking cone in function of jet-Pt
    const edm::Ref<pat::JetCollection> jetRef(jets, jet_idx - 1);
    fillTauFromJet(pfTauFromJet, reco::JetBaseRef(jetRef));

    // PATTau
    pat::Tau outputTauFromJet(pfTauFromJet);
    // Add tauIDs
    std::vector<pat::Tau::IdPair> newtauIds(tauIds_minimal.size() + tauIds_utag.size());
    for (size_t i = 0; i < tauIds_minimal.size(); ++i)
      newtauIds[i] = tauIds_minimal[i];
    for (size_t i = 0; i < tauIds_utag.size(); ++i)
      newtauIds[tauIds_minimal.size() + i] = tauIds_utag[i];
    outputTauFromJet.setTauIDs(newtauIds);
    // Add genTauJet match
    if (addGenJetMatch_) {
      reco::GenJetRef genJetTau = (*genJetMatch)[jetRef];
      if (genJetTau.isNonnull() && genJetTau.isAvailable()) {
        outputTauFromJet.setGenJet(genJetTau);
      }
    }
    outputTaus->push_back(outputTauFromJet);

  }  // end of jet loop

  // Taus non-matched to jets (usually at pt-threshold or/and eta boundaries)
  if (matched_taus.size() < inputTaus->size()) {
    for (size_t iTau = 0; iTau < inputTaus->size(); ++iTau) {
      if (matched_taus.count(iTau) > 0)
        continue;
      const pat::Tau& inputTau = inputTaus->at(iTau);
      pat::Tau outputTau(inputTau);
      const size_t nTauIds = inputTau.tauIDs().size();
      std::vector<pat::Tau::IdPair> tauIds(nTauIds + tauIds_utag.size());
      for (size_t i = 0; i < nTauIds; ++i)
        tauIds[i] = inputTau.tauIDs()[i];
      for (size_t i = 0; i < tauIds_utag.size(); ++i) {
        tauIds[nTauIds + i] = tauIds_utag[i];
        tauIds[nTauIds + i].second =
            (i != (size_t)tauId_utag_idx::ptcorr ? (i != (size_t)tauId_utag_idx::qconf ? -1 : 0) : 1);
      }
      outputTau.setTauIDs(tauIds);
      outputTaus->push_back(outputTau);
    }
  }  //non-matched taus

  // sort taus in pT
  std::sort(outputTaus->begin(), outputTaus->end(), pTTauComparator_);

  evt.put(std::move(outputTaus));
}

void PATTauHybridProducer::fillTauFromJet(reco::PFTau& pfTau, const reco::JetBaseRef& jet) {
  // Use tools as in PFTau builders to select tau decay products and isolation candidates
  typedef std::vector<reco::CandidatePtr> CandPtrs;

  // Get the charged hadron candidates
  CandPtrs pfChs, pfChsSig;
  // Check if we want to include electrons and muons in "charged hadron"
  // collection. This is the preferred behavior, as the PF lepton selections
  // are very loose.
  if (!usePFLeptonsAsChargedHadrons_) {
    pfChs = reco::tau::pfCandidatesByPdgId(*jet, 211);
  } else {
    pfChs = reco::tau::pfChargedCands(*jet);
  }
  // take 1st charged candidate with charge as of tau (collection is pt-sorted)
  if (pfTau.charge() == 0 || pfChs.size() == 1) {
    pfTau.setleadChargedHadrCand(pfChs[0]);
    pfTau.setleadCand(pfChs[0]);
    pfChsSig.push_back(pfChs[0]);
    pfChs.erase(pfChs.begin());
  } else {
    for (CandPtrs::iterator it = pfChs.begin(); it != pfChs.end();) {
      if ((*it)->charge() == pfTau.charge()) {
        pfTau.setleadChargedHadrCand(*it);
        pfTau.setleadCand(*it);
        pfChsSig.push_back(*it);
        it = pfChs.erase(it);
        break;
      } else {
        ++it;
      }
    }
    // In case of lack of candidate with charge same as of tau use leading charged candidate
    if (pfTau.leadChargedHadrCand().isNull() && !pfChs.empty()) {
      pfTau.setleadChargedHadrCand(pfChs[0]);
      pfTau.setleadCand(pfChs[0]);
      pfChsSig.push_back(pfChs[0]);
      pfChs.erase(pfChs.begin());
    }
  }
  // if more than one charged decay product is expected add all inside signal
  // cone around the leading track
  const double dR2Max = std::pow(pfTau.signalConeSize(), 2);
  if (pfTau.decayMode() >= reco::PFTau::kThreeProng0PiZero && pfTau.leadChargedHadrCand().isNonnull()) {
    for (CandPtrs::iterator it = pfChs.begin(); it != pfChs.end();) {
      if (deltaR2((*it)->p4(), pfTau.leadChargedHadrCand()->p4()) < dR2Max) {
        pfChsSig.push_back(*it);
        it = pfChs.erase(it);
      } else {
        ++it;
      }
    }
  }
  // Clean isolation candidates from low-pt and leptonic ones
  pfChs.erase(std::remove_if(pfChs.begin(),
                             pfChs.end(),
                             [](auto const& cand) { return cand->pt() < 0.5 || std::abs(cand->pdgId()) != 211; }),
              pfChs.end());
  // Set charged candidates
  pfTau.setsignalChargedHadrCands(pfChsSig);
  pfTau.setisolationChargedHadrCands(pfChs);

  // Get the gamma candidates (pi0 decay products)
  CandPtrs pfGammas, pfGammasSig;
  pfGammas = reco::tau::pfCandidatesByPdgId(*jet, 22);
  // In case of lack of leading charged candidate substiute it with leading gamma candidate
  if (pfTau.leadChargedHadrCand().isNull() && !pfGammas.empty()) {
    pfTau.setleadChargedHadrCand(pfGammas[0]);
    pfTau.setleadCand(pfGammas[0]);
    pfGammasSig.push_back(pfGammas[0]);
    pfGammas.erase(pfGammas.begin());
  }
  // Clean gamma candidates from low-pt ones
  pfGammas.erase(std::remove_if(pfGammas.begin(), pfGammas.end(), [](auto const& cand) { return cand->pt() < 0.5; }),
                 pfGammas.end());
  // if decay mode with pi0s is expected look for signal gamma candidates
  // within eta-phi strips around leading track
  if (pfTau.decayMode() % 5 != 0 && pfTau.leadChargedHadrCand().isNonnull()) {
    for (CandPtrs::iterator it = pfGammas.begin(); it != pfGammas.end();) {
      if (std::abs((*it)->eta() - pfTau.leadChargedHadrCand()->eta()) <
              std::clamp(0.2 * std::pow((*it)->pt(), -0.66), 0.05, 0.15) &&
          deltaPhi((*it)->phi(), pfTau.leadChargedHadrCand()->phi()) <
              std::clamp(0.35 * std::pow((*it)->pt(), -0.71), 0.05, 0.3)) {
        pfGammasSig.push_back(*it);
        it = pfGammas.erase(it);
      } else {
        ++it;
      }
    }
  }
  // Set gamma candidates
  pfTau.setsignalGammaCands(pfGammasSig);
  pfTau.setisolationGammaCands(pfGammas);
}

void PATTauHybridProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // patTauHybridProducer
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("slimmedTaus"));
  desc.add<edm::InputTag>("jetSource", edm::InputTag("slimmedJetsUpdated"));
  desc.add<double>("dRMax", 0.4);
  desc.add<double>("jetPtMin", 20.0);
  desc.add<double>("jetEtaMax", 2.5);
  desc.add<std::string>("UTagLabel", "pfParticleNetAK4JetTags");
  desc.add<std::string>("tagPrefix", "byUTag")->setComment("Prefix to be set for PUPPI or CHS tagger");
  desc.add<std::vector<std::string>>("UTagScoreNames", {})
      ->setComment("Output nodes for Unified Tagger (different for PNet vs UParT)");
  desc.add<std::string>("UtagPtCorrName", "ptcorr");
  desc.add<double>("tauScoreMin", -1)->setComment("Minimal value of the best tau score to built recovery tau");
  desc.add<double>("vsJetMin", -1)->setComment("Minimal value of UTag tau-vs-jet discriminant to built recovery tau");
  desc.add<bool>("checkTauScoreIsBest", false)
      ->setComment("If true, recovery tau is built only if one of tau scores is the highest");
  desc.add<double>("chargeAssignmentProbMin", 0.2)
      ->setComment("Minimal value of charge assignment probability to built recovery tau (0,0.5)");
  desc.add<bool>("addGenJetMatch", true)->setComment("add MC genTauJet matching");
  desc.add<edm::InputTag>("genJetMatch", edm::InputTag("tauGenJetMatch"));
  desc.add<bool>("usePFLeptonsAsChargedHadrons", true)
      ->setComment("If true, all charged particles are used as charged hadron candidates");

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTauHybridProducer);
