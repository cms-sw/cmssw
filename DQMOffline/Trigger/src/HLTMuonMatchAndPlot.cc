/** \file DQMOffline/Trigger/HLTMuonMatchAndPlot.cc
 *
 */

#include "DQMOffline/Trigger/interface/HLTMuonMatchAndPlot.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include <boost/algorithm/string/replace.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <utility>

//////////////////////////////////////////////////////////////////////////////
//////// Namespaces and Typedefs /////////////////////////////////////////////

using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;
using namespace l1extra;

using vstring = std::vector<std::string>;

//////////////////////////////////////////////////////////////////////////////
//////// HLTMuonMatchAndPlot Class Members ///////////////////////////////////

/// Constructor
HLTMuonMatchAndPlot::HLTMuonMatchAndPlot(const ParameterSet& pset, string hltPath, string moduleLabel, bool islastfilter)
    : hltProcessName_(pset.getParameter<string>("hltProcessName")),
      destination_(pset.getUntrackedParameter<string>("destination")),
      requiredTriggers_(pset.getUntrackedParameter<vstring>("requiredTriggers")),
      targetParams_(pset.getParameterSet("targetParams")),
      probeParams_(pset.getParameterSet("probeParams")),
      hltPath_(std::move(hltPath)),
      moduleLabel_(std::move(moduleLabel)),
      isLastFilter_(islastfilter),
      targetMuonEtaMax_(targetParams_.getUntrackedParameter<double>("recoMaxEtaCut", 0.)),
      targetMuonEtaMin_(targetParams_.getUntrackedParameter<double>("recoMinEtaCut", 0.)),
      targetIsMuonGlb_(targetParams_.getUntrackedParameter<bool>("recoGlbMuCut", false)),
      targetZ0Cut_(targetParams_.getUntrackedParameter<double>("z0Cut", 0.)),
      targetD0Cut_(targetParams_.getUntrackedParameter<double>("d0Cut", 0.)),
      targetptCutZ_(targetParams_.getUntrackedParameter<double>("ptCut_Z", 20.)),
      targetptCutJpsi_(targetParams_.getUntrackedParameter<double>("ptCut_Jpsi", 20.)),
      probeMuonEtaMax_(probeParams_.getUntrackedParameter<double>("recoMaxEtaCut", 0.)),
      probeMuonEtaMin_(probeParams_.getUntrackedParameter<double>("recoMinEtaCut", 0.)),
      probeIsMuonGlb_(probeParams_.getUntrackedParameter<bool>("recoGlbMuCut", false)),
      probeZ0Cut_(probeParams_.getUntrackedParameter<double>("z0Cut", 0.)),
      probeD0Cut_(probeParams_.getUntrackedParameter<double>("d0Cut", 0.)),
      triggerEtaMaxCut_(targetParams_.getUntrackedParameter<double>("hltMaxEtaCut", 0.)),
      triggerEtaMinCut_(targetParams_.getUntrackedParameter<double>("hltMinEtaCut", 0.)) {
  // Create std::map<string, T> from ParameterSets.
  fillMapFromPSet(binParams_, pset, "binParams");
  fillMapFromPSet(plotCuts_, pset, "plotCuts");

  // Get the trigger level.
  triggerLevel_ = "L3";
  TPRegexp levelRegexp("L[1-3]");
  //  size_t nModules = moduleLabels_.size();
  //  cout << moduleLabel_ << " " << hltPath_ << endl;
  TObjArray* levelArray = levelRegexp.MatchS(moduleLabel_);
  if (levelArray->GetEntriesFast() > 0) {
    triggerLevel_ = static_cast<const char*>(((TObjString*)levelArray->At(0))->GetString());
  }
  delete levelArray;

  // Get the pT cut by parsing the name of the HLT path.
  cutMinPt_ = 3;
  TPRegexp ptRegexp("Mu([0-9]*)");
  TObjArray* objArray = ptRegexp.MatchS(hltPath_);
  if (objArray->GetEntriesFast() >= 2) {
    auto* ptCutString = (TObjString*)objArray->At(1);
    cutMinPt_ = atoi(ptCutString->GetString());
    cutMinPt_ = ceil(cutMinPt_ * plotCuts_["minPtFactor"]);
  }
  delete objArray;
}

void HLTMuonMatchAndPlot::beginRun(DQMStore::IBooker& iBooker, const edm::Run& iRun, const edm::EventSetup& iSetup) {
  TPRegexp suffixPtCut("Mu[0-9]+$");

  string baseDir = destination_;
  if (baseDir[baseDir.size() - 1] != '/')
    baseDir += '/';
  string pathSansSuffix = hltPath_;
  if (hltPath_.rfind("_v") < hltPath_.length())
    pathSansSuffix = hltPath_.substr(0, hltPath_.rfind("_v"));

  if (isLastFilter_)
    iBooker.setCurrentFolder(baseDir + pathSansSuffix);
  else
    iBooker.setCurrentFolder(baseDir + pathSansSuffix + "/" + moduleLabel_);

  // Form is book1D(name, binningType, title) where 'binningType' is used
  // to fetch the bin settings from binParams_.

  for (const auto& suffix : EFFICIENCY_SUFFIXES) {
    if (isLastFilter_)
      iBooker.setCurrentFolder(baseDir + pathSansSuffix);
    else
      iBooker.setCurrentFolder(baseDir + pathSansSuffix + "/" + moduleLabel_);

    book1D(iBooker, "efficiencyEta_" + suffix, "eta", ";#eta;");
    book1D(iBooker, "efficiencyPhi_" + suffix, "phi", ";#phi;");
    book1D(iBooker, "efficiencyTurnOn_" + suffix, "pt", ";p_{T};");

    if (isLastFilter_)
      iBooker.setCurrentFolder(baseDir + pathSansSuffix);
    else
      iBooker.setCurrentFolder(baseDir + pathSansSuffix + "/" + moduleLabel_);

    if (!isLastFilter_)
      continue;  //this will be plotted only for the last filter

    book1D(iBooker, "efficiencyCharge_" + suffix, "charge", ";charge;");
  }
}

void HLTMuonMatchAndPlot::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {}

void HLTMuonMatchAndPlot::analyze(Handle<MuonCollection>& allMuons,
                                  Handle<BeamSpot>& beamSpot,
                                  Handle<VertexCollection>& vertices,
                                  Handle<TriggerEvent>& triggerSummary,
                                  Handle<TriggerResults>& triggerResults,
                                  const edm::TriggerNames& trigNames) {
  /*
  if(gen != 0) {
    for(g_part = gen->begin(); g_part != gen->end(); g_part++){
      if( abs(g_part->pdgId()) ==  13) {
        if(!( g_part->status() ==1 || (g_part->status() ==2 && abs(g_part->pdgId())==5))) continue;
        bool GenMomExists  (true);
        bool GenGrMomExists(true);
        if( g_part->pt() < 10.0 )  continue;
        if( fabs(g_part->eta()) > 2.4 ) continue;
        int gen_id= g_part->pdgId();
        const GenParticle* gen_lept = &(*g_part);
        // get mother of gen_lept
        const GenParticle* gen_mom = static_cast<const GenParticle*> (gen_lept->mother());
        if(gen_mom==NULL) GenMomExists=false;
        int m_id=-999;
        if(GenMomExists) m_id = gen_mom->pdgId();
        if(m_id != gen_id || !GenMomExists);
        else{
          int id= m_id;
          while(id == gen_id && GenMomExists){
            gen_mom = static_cast<const GenParticle*> (gen_mom->mother());
            if(gen_mom==NULL){
              GenMomExists=false;
            }
            if(GenMomExists) id=gen_mom->pdgId();
          }
        }
        if(GenMomExists) m_id = gen_mom->pdgId();
        gen_leptsp.push_back(gen_lept);
        gen_momsp.push_back(gen_mom);
      }
    }
  }
  std::vector<GenParticle> gen_lepts;
  for(size_t i = 0; i < gen_leptsp.size(); i++) {
    gen_lepts.push_back(*gen_leptsp[i]);
  }
  */

  // Select objects based on the configuration.
  MuonCollection targetMuons = selectedMuons(*allMuons, *beamSpot, true);
  MuonCollection probeMuons = selectedMuons(*allMuons, *beamSpot, false);
  TriggerObjectCollection allTriggerObjects = triggerSummary->getObjects();
  TriggerObjectCollection hltMuons = selectedTriggerObjects(allTriggerObjects, *triggerSummary);
  // Fill plots for HLT muons.

  // Find the best trigger object matches for the targetMuons.
  vector<size_t> matches = matchByDeltaR(targetMuons, hltMuons, plotCuts_[triggerLevel_ + "DeltaR"]);

  // Fill plots for matched muons.
  for (size_t i = 0; i < targetMuons.size(); i++) {
    Muon& muon = targetMuons[i];

    // Fill numerators and denominators for efficiency plots.
    for (const auto& suffix : EFFICIENCY_SUFFIXES) {
      // If no match was found, then the numerator plots don't get filled.
      if (suffix == "numer" && matches[i] >= targetMuons.size())
        continue;

      if (muon.pt() > cutMinPt_) {
        hists_["efficiencyEta_" + suffix]->Fill(muon.eta());
      }

      if (fabs(muon.eta()) < plotCuts_["maxEta"]) {
        hists_["efficiencyTurnOn_" + suffix]->Fill(muon.pt());
      }

      if (muon.pt() > cutMinPt_ && fabs(muon.eta()) < plotCuts_["maxEta"]) {
        const Track* track = nullptr;
        if (muon.isTrackerMuon())
          track = &*muon.innerTrack();
        else if (muon.isStandAloneMuon())
          track = &*muon.outerTrack();
        if (track) {
          hists_["efficiencyPhi_" + suffix]->Fill(muon.phi());

          if (isLastFilter_) {
            hists_["efficiencyCharge_" + suffix]->Fill(muon.charge());
          }
        }
      }
    }  // finish loop numerator / denominator...

    if (!isLastFilter_)
      continue;
    // Fill plots for tag and probe
    // Muon cannot be a tag because doesn't match an hlt muon
    if (matches[i] >= targetMuons.size())
      continue;
  }  // End loop over targetMuons.

  // fill eff histograms for reference trigger method
  // Denominator: events passing reference trigger and two target muons
  // Numerator:   events in the denominator with two target muons
  // matched to hlt muons
  if (!isLastFilter_)
    return;
  unsigned int numTriggers = trigNames.size();
  bool passTrigger = false;
  if (requiredTriggers_.empty())
    passTrigger = true;
  for (auto const& requiredTrigger : requiredTriggers_) {
    for (unsigned int hltIndex = 0; hltIndex < numTriggers; ++hltIndex) {
      passTrigger = (trigNames.triggerName(hltIndex).find(requiredTrigger) != std::string::npos &&
                     triggerResults->wasrun(hltIndex) && triggerResults->accept(hltIndex));
      if (passTrigger)
        break;
    }
  }

  int nMatched = 0;
  for (unsigned long matche : matches) {
    if (matche < targetMuons.size())
      nMatched++;
  }

  string nonSameSignPath = hltPath_;
  bool ssPath = false;
  if (nonSameSignPath.rfind("_SameSign") < nonSameSignPath.length()) {
    ssPath = true;
    nonSameSignPath = boost::replace_all_copy<string>(nonSameSignPath, "_SameSign", "");
    nonSameSignPath = nonSameSignPath.substr(0, nonSameSignPath.rfind("_v") + 2);
  }
  bool passTriggerSS = false;
  if (ssPath) {
    for (unsigned int hltIndex = 0; hltIndex < numTriggers; ++hltIndex) {
      passTriggerSS =
          passTriggerSS || (trigNames.triggerName(hltIndex).substr(0, nonSameSignPath.size()) == nonSameSignPath &&
                            triggerResults->wasrun(hltIndex) && triggerResults->accept(hltIndex));
    }
  }

  string nonDZPath = hltPath_;
  bool dzPath = false;
  if (nonDZPath.rfind("_DZ") < nonDZPath.length()) {
    dzPath = true;
    nonDZPath = boost::replace_all_copy<string>(nonDZPath, "_DZ", "");
    nonDZPath = nonDZPath.substr(0, nonDZPath.rfind("_v") + 2);
  }
  bool passTriggerDZ = false;
  if (dzPath) {
    for (unsigned int hltIndex = 0; hltIndex < numTriggers; ++hltIndex) {
      passTriggerDZ = passTriggerDZ || (trigNames.triggerName(hltIndex).find(nonDZPath) != std::string::npos &&
                                        triggerResults->wasrun(hltIndex) && triggerResults->accept(hltIndex));
    }
  }

}  // End analyze() method.

// Method to fill binning parameters from a vector of doubles.
void HLTMuonMatchAndPlot::fillEdges(size_t& nBins, float*& edges, const vector<double>& binning) {
  if (binning.size() < 3) {
    LogWarning("HLTMuonVal") << "Invalid binning parameters!";
    return;
  }

  // Fixed-width binning.
  if (binning.size() == 3) {
    nBins = binning[0];
    edges = new float[nBins + 1];
    const double min = binning[1];
    const double binwidth = (binning[2] - binning[1]) / nBins;
    for (size_t i = 0; i <= nBins; i++)
      edges[i] = min + (binwidth * i);
  }

  // Variable-width binning.
  else {
    nBins = binning.size() - 1;
    edges = new float[nBins + 1];
    for (size_t i = 0; i <= nBins; i++)
      edges[i] = binning[i];
  }
}

// This is an unorthodox method of getting parameters, but cleaner in my mind
// It looks for parameter 'target' in the pset, and assumes that all entries
// have the same class (T), filling the values into a std::map.
template <class T>
void HLTMuonMatchAndPlot::fillMapFromPSet(map<string, T>& m, const ParameterSet& pset, const string& target) {
  // Get the ParameterSet with name 'target' from 'pset'
  ParameterSet targetPset;
  if (pset.existsAs<ParameterSet>(target, true))  // target is tracked
    targetPset = pset.getParameterSet(target);
  else if (pset.existsAs<ParameterSet>(target, false))  // target is untracked
    targetPset = pset.getUntrackedParameterSet(target);

  // Get the parameter names from targetPset
  vector<string> names = targetPset.getParameterNames();
  vector<string>::const_iterator iter;

  for (iter = names.begin(); iter != names.end(); ++iter) {
    if (targetPset.existsAs<T>(*iter, true))  // target is tracked
      m[*iter] = targetPset.getParameter<T>(*iter);
    else if (targetPset.existsAs<T>(*iter, false))  // target is untracked
      m[*iter] = targetPset.getUntrackedParameter<T>(*iter);
  }
}

// A generic method to find the best deltaR matches from 2 collections.
template <class T1, class T2>
vector<size_t> HLTMuonMatchAndPlot::matchByDeltaR(const vector<T1>& collection1,
                                                  const vector<T2>& collection2,
                                                  const double maxDeltaR) {
  const size_t n1 = collection1.size();
  const size_t n2 = collection2.size();

  vector<size_t> result(n1, -1);
  vector<vector<double> > deltaRMatrix(n1, vector<double>(n2, NOMATCH));

  for (size_t i = 0; i < n1; i++)
    for (size_t j = 0; j < n2; j++) {
      deltaRMatrix[i][j] = deltaR(collection1[i], collection2[j]);
    }

  // Run through the matrix n1 times to make sure we've found all matches.
  for (size_t k = 0; k < n1; k++) {
    size_t i_min = -1;
    size_t j_min = -1;
    double minDeltaR = maxDeltaR;
    // find the smallest deltaR
    for (size_t i = 0; i < n1; i++)
      for (size_t j = 0; j < n2; j++)
        if (deltaRMatrix[i][j] < minDeltaR) {
          i_min = i;
          j_min = j;
          minDeltaR = deltaRMatrix[i][j];
        }
    // If a match has been made, save it and make those candidates unavailable.
    if (minDeltaR < maxDeltaR) {
      result[i_min] = j_min;
      deltaRMatrix[i_min] = vector<double>(n2, NOMATCH);
      for (size_t i = 0; i < n1; i++)
        deltaRMatrix[i][j_min] = NOMATCH;
    }
  }

  return result;
}

MuonCollection HLTMuonMatchAndPlot::selectedMuons(const MuonCollection& allMuons,
                                                  const BeamSpot& beamSpot,
                                                  bool isTargetMuons) {
  MuonCollection reducedMuons;
  double RecoMuonEtaMax = isTargetMuons ? targetMuonEtaMax_ : probeMuonEtaMax_;
  double RecoMuonEtaMin = isTargetMuons ? targetMuonEtaMin_ : probeMuonEtaMin_;
  bool IsMuonGlb = isTargetMuons ? targetIsMuonGlb_ : probeIsMuonGlb_;
  double d0Cut = isTargetMuons ? targetD0Cut_ : probeD0Cut_;
  double z0Cut = isTargetMuons ? targetZ0Cut_ : probeZ0Cut_;

  for (auto const& mu : allMuons) {
    const Track* track = nullptr;
    if (mu.isTrackerMuon())
      track = &*mu.innerTrack();
    else if (mu.isStandAloneMuon())
      track = &*mu.outerTrack();
    // minimun ID (requested for cosmics) is being a StandAlone muon
    bool muID = IsMuonGlb ? mu.isGlobalMuon() : mu.isStandAloneMuon();
    if (track && muID && abs(mu.eta()) < RecoMuonEtaMax && abs(mu.eta()) >= RecoMuonEtaMin &&
        fabs(track->dxy(beamSpot.position())) < d0Cut && fabs(track->dz(beamSpot.position())) < z0Cut)
      reducedMuons.push_back(mu);
  }

  return reducedMuons;
}

TriggerObjectCollection HLTMuonMatchAndPlot::selectedTriggerObjects(const TriggerObjectCollection& triggerObjects,
                                                                    const TriggerEvent& triggerSummary) {
  InputTag filterTag(moduleLabel_, "", hltProcessName_);
  size_t filterIndex = triggerSummary.filterIndex(filterTag);

  TriggerObjectCollection selectedObjects;
  if (filterIndex < triggerSummary.sizeFilters()) {
    const Keys& keys = triggerSummary.filterKeys(filterIndex);
    for (unsigned short key : keys) {
      TriggerObject foundObject = triggerObjects[key];
      if (abs(foundObject.eta()) < triggerEtaMaxCut_ && abs(foundObject.eta()) >= triggerEtaMinCut_)
        selectedObjects.push_back(foundObject);
    }
  }

  return selectedObjects;
}

void HLTMuonMatchAndPlot::book1D(DQMStore::IBooker& iBooker, string name, const string& binningType, string title) {
  /* Properly delete the array of floats that has been allocated on
   * the heap by fillEdges.  Avoid multiple copies and internal ROOT
   * clones by simply creating the histograms directly in the DQMStore
   * using the appropriate book1D method to handle the variable bins
   * case. */

  size_t nBins;
  float* edges = nullptr;
  fillEdges(nBins, edges, binParams_[binningType]);
  hists_[name] = iBooker.book1D(name, title, nBins, edges);
  if (hists_[name])
    if (hists_[name]->getTH1F()->GetSumw2N())
      hists_[name]->enableSumw2();

  if (edges)
    delete[] edges;
}

void HLTMuonMatchAndPlot::book2D(DQMStore::IBooker& iBooker,
                                 const string& name,
                                 const string& binningTypeX,
                                 const string& binningTypeY,
                                 const string& title) {
  /* Properly delete the arrays of floats that have been allocated on
   * the heap by fillEdges.  Avoid multiple copies and internal ROOT
   * clones by simply creating the histograms directly in the DQMStore
   * using the appropriate book2D method to handle the variable bins
   * case. */

  size_t nBinsX;
  float* edgesX = nullptr;
  fillEdges(nBinsX, edgesX, binParams_[binningTypeX]);

  size_t nBinsY;
  float* edgesY = nullptr;
  fillEdges(nBinsY, edgesY, binParams_[binningTypeY]);

  hists_[name] = iBooker.book2D(name.c_str(), title.c_str(), nBinsX, edgesX, nBinsY, edgesY);
  if (hists_[name])
    if (hists_[name]->getTH2F()->GetSumw2N())
      hists_[name]->enableSumw2();

  if (edgesX)
    delete[] edgesX;
  if (edgesY)
    delete[] edgesY;
}
