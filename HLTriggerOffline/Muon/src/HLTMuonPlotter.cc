#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/path_configuration.h"
#include "HLTriggerOffline/Muon/interface/HLTMuonPlotter.h"

#include <algorithm>

#include "TPRegexp.h"
#include "TObjArray.h"
#include "TObjString.h"

namespace {
  const unsigned int kNull = (unsigned int)-1;
}

HLTMuonPlotter::HLTMuonPlotter(const edm::ParameterSet &pset,
                               const std::string &hltPath,
                               const std::vector<std::string> &moduleLabels,
                               const std::vector<std::string> &stepLabels,
                               const edm::EDGetTokenT<trigger::TriggerEventWithRefs> &triggerEventWithRefsToken,
                               const edm::EDGetTokenT<reco::GenParticleCollection> &genParticlesToken,
                               const edm::EDGetTokenT<reco::MuonCollection> &recoMuonsToken,
                               const L1MuonMatcherAlgoForDQM &l1Matcher)
    : hltPath_(hltPath),
      hltProcessName_(pset.getParameter<std::string>("hltProcessName")),
      moduleLabels_(moduleLabels),
      stepLabels_(stepLabels),
      triggerEventWithRefsToken_(triggerEventWithRefsToken),
      genParticleToken_(genParticlesToken),
      recMuonToken_(recoMuonsToken),
      genMuonSelector_(pset.getParameter<std::string>("genMuonCut")),
      recMuonSelector_(pset.getParameter<std::string>("recMuonCut")),
      cutsDr_(pset.getParameter<std::vector<double>>("cutsDr")),
      parametersEta_(pset.getParameter<std::vector<double>>("parametersEta")),
      parametersPhi_(pset.getParameter<std::vector<double>>("parametersPhi")),
      parametersTurnOn_(pset.getParameter<std::vector<double>>("parametersTurnOn")),
      l1Matcher_(l1Matcher),
      isInvalid_(false) {
  if (moduleLabels_.empty()) {
    edm::LogError("HLTMuonPlotter") << "Invalid inputs: 'moduleLabels_' is empty."
                                    << "\nMonitorElements for HLT path '" << hltPath_ << "' will not be produced.";
    isInvalid_ = true;
  } else if (stepLabels_.size() != moduleLabels_.size() + 1) {
    edm::LogError err("HLTMuonPlotter");
    err << "Invalid inputs: 'stepLabels_.size()' must equal 'moduleLabels_.size() + 1'.";
    err << "\nMonitorElements for HLT path '" << hltPath_ << "' will not be produced.";
    err << "\n  stepLabels_ = (";
    for (auto const &foo : stepLabels_)
      err << " " << foo;
    err << " )";
    err << "\n  moduleLabels_ = (";
    for (auto const &foo : moduleLabels_)
      err << " " << foo;
    err << " )";
    isInvalid_ = true;
  }
}

void HLTMuonPlotter::beginRun(DQMStore::IBooker &iBooker, const edm::Run &iRun, const edm::EventSetup &iSetup) {
  if (isInvalid_)
    return;

  l1Matcher_.init(iSetup);

  cutMaxEta_ = 2.4;
  if (hltPath_.find("eta2p1") != std::string::npos)
    cutMaxEta_ = 2.1;

  // Choose a pT cut for gen/rec muons based on the pT cut in the hltPath_
  unsigned int threshold = 0;
  TPRegexp ptRegexp("Mu([0-9]+)");
  TObjArray *regexArray = ptRegexp.MatchS(hltPath_);
  if (regexArray->GetEntriesFast() == 2) {
    threshold = atoi(((TObjString *)regexArray->At(1))->GetString());
  }
  delete regexArray;
  // We select a whole number min pT cut slightly above the hltPath_'s final
  // pt threshold, then subtract a bit to let through particle gun muons with
  // exact integer pT:
  cutMinPt_ = ceil(threshold * 1.1) - 0.01;
  if (cutMinPt_ < 0.)
    cutMinPt_ = 0.;

  std::string baseDir = "HLT/Muon/Distributions/";
  iBooker.setCurrentFolder(baseDir + hltPath_);

  std::vector<std::string> sources(2);
  sources[0] = "gen";
  sources[1] = "rec";

  elements_["CutMinPt"] = iBooker.bookFloat("CutMinPt");
  elements_["CutMaxEta"] = iBooker.bookFloat("CutMaxEta");
  elements_["CutMinPt"]->Fill(cutMinPt_);
  elements_["CutMaxEta"]->Fill(cutMaxEta_);

  for (size_t i = 0; i < sources.size(); i++) {
    std::string source = sources[i];
    for (size_t j = 0; j < stepLabels_.size(); j++) {
      bookHist(iBooker, hltPath_, stepLabels_[j], source, "Eta");
      bookHist(iBooker, hltPath_, stepLabels_[j], source, "Phi");
      bookHist(iBooker, hltPath_, stepLabels_[j], source, "MaxPt1");
      bookHist(iBooker, hltPath_, stepLabels_[j], source, "MaxPt2");
    }
  }
}

void HLTMuonPlotter::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  if (isInvalid_)
    return;

  auto const triggerEventWithRefsHandle = iEvent.getHandle(triggerEventWithRefsToken_);
  if (triggerEventWithRefsHandle.failedToGet()) {
    edm::LogError("HLTMuonPlotter") << "No trigger summary found";
    return;
  }

  auto const recoMuonsHandle = iEvent.getHandle(recMuonToken_);
  auto const genParticlesHandle = iEvent.getHandle(genParticleToken_);

  LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter] --------------";
  LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter] Event: " << iEvent.id();

  const int nFilters = moduleLabels_.size();
  const int nSteps = stepLabels_.size();
  const bool hasStepL1 = (stepLabels_.size() > 1 and stepLabels_[1] == "L1");
  const int nStepsHlt = hasStepL1 ? nSteps - 2 : nSteps - 1;
  const bool isDoubleMuonPath = (hltPath_.find("Double") != std::string::npos);
  const int nObjectsToPassPath = (isDoubleMuonPath) ? 2 : 1;

  LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter] HLTPath=" << hltPath_ << " nFilters=" << nFilters
                             << " nSteps=" << nSteps << " hasStepL1=" << hasStepL1 << " nStepsHlt=" << nStepsHlt
                             << " isDoubleMuonPath=" << isDoubleMuonPath
                             << " nObjectsToPassPath=" << nObjectsToPassPath;

  if (nFilters + 1 == nSteps) {
    LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]   stepLabels | moduleLabels";
    for (int istep = 0; istep < nSteps; ++istep) {
      if (istep == 0)
        LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]   " << stepLabels_[istep] << " | [N/A]";
      else
        LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]   " << stepLabels_[istep] << " | " << moduleLabels_[istep - 1];
    }
  }

  std::vector<std::string> sources;
  if (genParticlesHandle.isValid())
    sources.push_back("gen");
  if (recoMuonsHandle.isValid())
    sources.push_back("rec");

  for (size_t sourceNo = 0; sourceNo < sources.size(); sourceNo++) {
    std::string const &source = sources[sourceNo];

    LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter] source=" << source;

    // Make each good gen/rec muon into the base cand for a MatchStruct
    std::vector<MatchStruct> matches;

    if (source == "gen") {
      if (genParticlesHandle.isValid()) {
        matches.reserve(matches.size() + genParticlesHandle->size());
        for (auto const &genp : *genParticlesHandle)
          if (genMuonSelector_(genp))
            matches.emplace_back(MatchStruct(&genp));
      }
    } else if (source == "rec") {
      if (recoMuonsHandle.isValid()) {
        matches.reserve(matches.size() + recoMuonsHandle->size());
        for (auto const &recomu : *recoMuonsHandle)
          if (recMuonSelector_(recomu))
            matches.emplace_back(MatchStruct(&recomu));
      }
    }

    // Sort the MatchStructs by pT for later filling of turn-on curve
    std::sort(matches.begin(), matches.end(), matchesByDescendingPt());

    l1t::MuonVectorRef candsL1;
    std::vector<std::vector<reco::RecoChargedCandidateRef>> refsHlt(nStepsHlt);
    std::vector<std::vector<const reco::RecoChargedCandidate *>> candsHlt(nStepsHlt);

    for (int idx = 0; idx < nFilters; ++idx) {
      auto const moduleLabelStripped =
          edm::path_configuration::removeSchedulingTokensFromModuleLabel(moduleLabels_[idx]);
      auto const iTag = edm::InputTag(moduleLabelStripped, "", hltProcessName_);
      auto const iFilter = triggerEventWithRefsHandle->filterIndex(iTag);
      auto const iFilterValid = (iFilter < triggerEventWithRefsHandle->size());

      LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]   InputTag[" << idx << "]: " << moduleLabels_[idx]
                                 << "::" << hltProcessName_ << " (filterIndex = " << iFilter
                                 << ", valid = " << iFilterValid << ")";

      if (iFilterValid) {
        if (idx == 0 and hasStepL1)
          triggerEventWithRefsHandle->getObjects(iFilter, trigger::TriggerL1Mu, candsL1);
        else {
          auto const hltStep = hasStepL1 ? idx - 1 : idx;
          triggerEventWithRefsHandle->getObjects(iFilter, trigger::TriggerMuon, refsHlt[hltStep]);
        }
      } else
        LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]    No collection with " << iTag;
    }

    for (int i = 0; i < nStepsHlt; i++) {
      for (size_t j = 0; j < refsHlt[i].size(); j++) {
        if (refsHlt[i][j].isAvailable())
          candsHlt[i].push_back(&*refsHlt[i][j]);
        else
          edm::LogWarning("HLTMuonPlotter") << "Ref refsHlt[i][j]: product not available " << i << " " << j;
      }
    }

    // Add trigger objects to the MatchStructs
    findMatches(matches, candsL1, candsHlt);

    LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]   Number of Candidates = " << matches.size();

    for (auto const &match_i : matches) {
      if (!match_i.candBase)
        continue;
      LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]    CandBase: pt=" << match_i.candBase->pt()
                                 << " eta=" << match_i.candBase->eta() << " phi=" << match_i.candBase->phi();
      if (match_i.candL1)
        LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]      CandL1: pt=" << match_i.candL1->pt()
                                   << " eta=" << match_i.candL1->eta() << " phi=" << match_i.candL1->phi();
      else
        LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]      CandL1: NULL";

      int ihlt = -1;
      for (auto const *chlt : match_i.candHlt) {
        ++ihlt;
        if (chlt)
          LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]      CandHLT[" << ihlt << "]: pt=" << chlt->pt()
                                     << " eta=" << chlt->eta() << " phi=" << chlt->phi();
        else
          LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]      CandHLT[" << ihlt << "]: NULL";
      }
    }

    std::vector<size_t> matchesInEtaRange;
    std::vector<bool> hasMatch(matches.size(), true);

    for (int step = 0; step < nSteps; step++) {
      int const hltStep = hasStepL1 ? step - 2 : step - 1;
      size_t level = 0;
      if ((stepLabels_[step].find("L3TkIso") != std::string::npos) ||
          (stepLabels_[step].find("TkTkIso") != std::string::npos))
        level = 6;
      else if ((stepLabels_[step].find("L3HcalIso") != std::string::npos) ||
               (stepLabels_[step].find("TkEcalIso") != std::string::npos))
        level = 5;
      else if ((stepLabels_[step].find("L3EcalIso") != std::string::npos) ||
               (stepLabels_[step].find("TkEcalIso") != std::string::npos))
        level = 4;
      else if ((stepLabels_[step].find("L3") != std::string::npos) ||
               (stepLabels_[step].find("Tk") != std::string::npos))
        level = 3;
      else if (stepLabels_[step].find("L2") != std::string::npos)
        level = 2;
      else if (stepLabels_[step].find("L1") != std::string::npos)
        level = 1;

      for (size_t j = 0; j < matches.size(); j++) {
        if (level == 0) {
          if (std::abs(matches[j].candBase->eta()) < cutMaxEta_)
            matchesInEtaRange.push_back(j);
        } else if (level == 1) {
          if (matches[j].candL1 == nullptr)
            hasMatch[j] = false;
        } else if (level >= 2) {
          if (matches[j].candHlt.at(hltStep) == nullptr)
            hasMatch[j] = false;
          else if (!hasMatch[j]) {
            LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]     match found for " << source << " candidate " << j
                                       << " in HLT step " << hltStep << " of " << nStepsHlt
                                       << " without previous match!";
            break;
          }
        }
      }

      LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]    (step=" << step << ", level=" << level
                                 << ", hltStep=" << hltStep << ") matchesInEtaRange: [ "
                                 << this->vector_to_string(matchesInEtaRange) << " ]";

      LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]    (step=" << step << ", level=" << level
                                 << ", hltStep=" << hltStep << ") hasMatch: [ " << this->vector_to_string(hasMatch)
                                 << " ]";

      if (std::count(hasMatch.begin(), hasMatch.end(), true) < nObjectsToPassPath)
        break;

      std::string const pre = source + "Pass";
      std::string const post = "_" + stepLabels_[step];

      for (size_t j = 0; j < matches.size(); j++) {
        float const pt = matches[j].candBase->pt();
        float const eta = matches[j].candBase->eta();
        float const phi = matches[j].candBase->phi();
        if (hasMatch[j]) {
          if (!matchesInEtaRange.empty() && j == matchesInEtaRange[0]) {
            elements_[pre + "MaxPt1" + post]->Fill(pt);
            LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]     FILL(" << pre + "MaxPt1" + post << ") value = " << pt;
          }
          if (matchesInEtaRange.size() >= 2 && j == matchesInEtaRange[1]) {
            elements_[pre + "MaxPt2" + post]->Fill(pt);
            LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]     FILL(" << pre + "MaxPt2" + post << ") value = " << pt;
          }
          if (pt > cutMinPt_) {
            elements_[pre + "Eta" + post]->Fill(eta);
            LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]     FILL(" << pre + "Eta" + post << ") value = " << eta;
            if (std::abs(eta) < cutMaxEta_) {
              elements_[pre + "Phi" + post]->Fill(phi);
              LogTrace("HLTMuonPlotter") << "[HLTMuonPlotter]     FILL(" << pre + "Phi" + post << ") value = " << phi;
            }
          }
        }
      }
    }
  }  // End loop over sources
}

void HLTMuonPlotter::findMatches(std::vector<MatchStruct> &matches,
                                 const l1t::MuonVectorRef &candsL1,
                                 const std::vector<std::vector<const reco::RecoChargedCandidate *>> &candsHlt) {
  std::set<size_t>::iterator it;

  std::set<size_t> indicesL1;
  for (size_t i = 0; i < candsL1.size(); i++)
    indicesL1.insert(i);

  std::vector<set<size_t>> indicesHlt(candsHlt.size());
  for (size_t i = 0; i < candsHlt.size(); i++)
    for (size_t j = 0; j < candsHlt[i].size(); j++)
      indicesHlt[i].insert(j);

  for (size_t i = 0; i < matches.size(); i++) {
    const reco::Candidate *cand = matches[i].candBase;

    double bestDeltaR = cutsDr_[0];
    size_t bestMatch = kNull;
    for (it = indicesL1.begin(); it != indicesL1.end(); it++) {
      if (candsL1[*it].isAvailable()) {
        double dR = deltaR(cand->eta(), cand->phi(), candsL1[*it]->eta(), candsL1[*it]->phi());
        if (dR < bestDeltaR) {
          bestMatch = *it;
          bestDeltaR = dR;
        }
        // TrajectoryStateOnSurface propagated;
        // float dR = 999., dPhi = 999.;
        // bool isValid = l1Matcher_.match(* cand, * candsL1[*it],
        //                                 dR, dPhi, propagated);
        // if (isValid && dR < bestDeltaR) {
        //   bestMatch = *it;
        //   bestDeltaR = dR;
        // }
      } else {
        edm::LogWarning("HLTMuonPlotter") << "Ref candsL1[*it]: product not available " << *it;
      }
    }

    if (bestMatch != kNull)
      matches[i].candL1 = &*candsL1[bestMatch];
    indicesL1.erase(bestMatch);

    matches[i].candHlt.assign(candsHlt.size(), nullptr);
    for (size_t j = 0; j < candsHlt.size(); j++) {
      size_t level = (candsHlt.size() == 4) ? (j < 2) ? 2 : 3 : (candsHlt.size() == 2) ? (j < 1) ? 2 : 3 : 2;
      bestDeltaR = cutsDr_[level - 2];
      bestMatch = kNull;
      for (it = indicesHlt[j].begin(); it != indicesHlt[j].end(); it++) {
        double dR = deltaR(cand->eta(), cand->phi(), candsHlt[j][*it]->eta(), candsHlt[j][*it]->phi());
        if (dR < bestDeltaR) {
          bestMatch = *it;
          bestDeltaR = dR;
        }
      }
      if (bestMatch != kNull)
        matches[i].candHlt[j] = candsHlt[j][bestMatch];
      indicesHlt[j].erase(bestMatch);
    }
  }
}

void HLTMuonPlotter::bookHist(DQMStore::IBooker &iBooker,
                              std::string const &path,
                              std::string const &label,
                              std::string const &source,
                              std::string const &type) {
  std::string sourceUpper = source;
  sourceUpper[0] = toupper(sourceUpper[0]);
  std::string name = source + "Pass" + type + "_" + label;
  TH1F *h;

  if (type.find("MaxPt") != std::string::npos) {
    std::string desc = (type == "MaxPt1") ? "Leading" : "Next-to-Leading";
    std::string title = "pT of " + desc + " " + sourceUpper + " Muon " + "matched to " + label;
    const size_t nBins = parametersTurnOn_.size() - 1;
    float *edges = new float[nBins + 1];
    for (size_t i = 0; i < nBins + 1; i++)
      edges[i] = parametersTurnOn_[i];
    h = new TH1F(name.c_str(), title.c_str(), nBins, edges);
  } else {
    std::string symbol = (type == "Eta") ? "#eta" : "#phi";
    std::string title = symbol + " of " + sourceUpper + " Muons " + "matched to " + label;
    std::vector<double> params = (type == "Eta") ? parametersEta_ : parametersPhi_;
    int nBins = (int)params[0];
    double min = params[1];
    double max = params[2];
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  }

  h->Sumw2();
  elements_[name] = iBooker.book1D(name, h);
  delete h;
}
