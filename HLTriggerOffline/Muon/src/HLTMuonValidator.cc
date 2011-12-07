/** \file HLTMuonValidator.cc
 *  $Date: 2010/11/04 12:56:22 $
 *  $Revision: 1.23 $
 */



#include "HLTriggerOffline/Muon/interface/HLTMuonValidator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"



using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;
using namespace l1extra;



typedef vector<ParameterSet> Parameters;



HLTMuonValidator::HLTMuonValidator(const ParameterSet & pset) :
  l1Matcher_(pset)
{

  hltProcessName_  = pset.getParameter< string         >("hltProcessName");
  hltPathsToCheck_ = pset.getParameter< vector<string> >("hltPathsToCheck");

  genParticleLabel_ = pset.getParameter<string>("genParticleLabel");
      recMuonLabel_ = pset.getParameter<string>(    "recMuonLabel");
       l1CandLabel_ = pset.getParameter<string>(     "l1CandLabel");
       l2CandLabel_ = pset.getParameter<string>(     "l2CandLabel");
       l3CandLabel_ = pset.getParameter<string>(     "l3CandLabel");

  cutsDr_      = pset.getParameter< vector<double> >("cutsDr"     );

  parametersEta_    = pset.getParameter< vector<double> >("parametersEta");
  parametersPhi_    = pset.getParameter< vector<double> >("parametersPhi");
  parametersTurnOn_ = pset.getParameter< vector<double> >("parametersTurnOn");

  genMuonCut_ = pset.getParameter<string>("genMuonCut");
  recMuonCut_ = pset.getParameter<string>("recMuonCut");

  genMuonSelector_ = 0;
  recMuonSelector_ = 0;

  dbe_ = Service<DQMStore>().operator->();
  dbe_->setVerbose(0);

}



void 
HLTMuonValidator::beginJob() 
{
}



void 
HLTMuonValidator::beginRun(const Run & iRun, const EventSetup & iSetup) 
{

  static int runNumber = 0;
  runNumber++;

  l1Matcher_.init(iSetup);

  bool changedConfig;
  if (!hltConfig_.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    LogError("HLTMuonVal") << "Initialization of HLTConfigProvider failed!!"; 
    return;
  }

  hltPaths_.clear();
  filterLabels_.clear();
  vector<string> validTriggerNames = hltConfig_.triggerNames();
  for (size_t i = 0; i < hltPathsToCheck_.size(); i++) {
    TPRegexp pattern(hltPathsToCheck_[i]);
    for (size_t j = 0; j < validTriggerNames.size(); j++)
      if (TString(validTriggerNames[j]).Contains(pattern))
        hltPaths_.insert(validTriggerNames[j]);
    // Add fake HLT path where we bypass all filters
    if (TString("NoFilters").Contains(pattern))
      hltPaths_.insert("NoFilters");
  }

  initializeHists();

}



void
HLTMuonValidator::initializeHists()
{

  vector<string> sources(2);
  sources[0] = "gen";
  sources[1] = "rec";

  set<string>::iterator iPath;
  TPRegexp suffixPtCut("[0-9]+$");

  for (iPath = hltPaths_.begin(); iPath != hltPaths_.end(); iPath++) {
 
    string path = * iPath;

    if (path == "NoFilters") {
      filterLabels_[path].push_back("hltL1sL1SingleMuOpenL1SingleMu0");
      // The L2 and L3 collections will be built manually later on
      filterLabels_[path].push_back("");
      filterLabels_[path].push_back("");
    }

    vector<string> moduleLabels;
    if (path != "NoFilters") moduleLabels = hltConfig_.moduleLabels(path);

    for (size_t i = 0; i < moduleLabels.size(); i++)
      if (moduleLabels[i].find("Filtered") != string::npos)
        filterLabels_[path].push_back(moduleLabels[i]);

    // Getting a reliable L1 pT measurement is impossible beyond 2.1, so most
    // paths have no L1 beyond 2.1 except those passing kLooseL1Requirement
    double cutMaxEta = (TString(path).Contains(kLooseL1Requirement)) ? 2.4:2.1;

    // Choose a pT cut for gen/rec muons based on the pT cut in the path
    unsigned int index = TString(path).Index(suffixPtCut);
    unsigned int threshold = 3;
    if (index < path.length()) threshold = atoi(path.substr(index).c_str());
    // We select a whole number min pT cut slightly above the path's final 
    // pt threshold, then subtract a bit to let through particle gun muons with
    // exact integer pT:
    double cutMinPt = ceil(threshold * 1.1) - 0.01;
    if (cutMinPt < 0. || path == "NoFilters") cutMinPt = 0.;
    cutsMinPt_[path] = cutMinPt;

    string baseDir = "HLT/Muon/Distributions/";
    dbe_->setCurrentFolder(baseDir + path);

    if (dbe_->get(baseDir + path + "/CutMinPt") == 0) {

      elements_[path + "_" + "CutMinPt" ] = dbe_->bookFloat("CutMinPt" );
      elements_[path + "_" + "CutMaxEta"] = dbe_->bookFloat("CutMaxEta");
      elements_[path + "_" + "CutMinPt" ]->Fill(cutMinPt);
      elements_[path + "_" + "CutMaxEta"]->Fill(cutMaxEta);

      // Standardize the names that will be applied to each step
      const int nFilters = filterLabels_[path].size();
      stepLabels_[path].push_back("All");
      stepLabels_[path].push_back("L1");
      if (nFilters == 2) {
        stepLabels_[path].push_back("L2");
      }
      if (nFilters == 3) {
        stepLabels_[path].push_back("L2");
        stepLabels_[path].push_back("L3");
      }
      if (nFilters == 5) {
        stepLabels_[path].push_back("L2");
        stepLabels_[path].push_back("L2Iso");
        stepLabels_[path].push_back("L3");
        stepLabels_[path].push_back("L3Iso");
      }

      //     string l1Name = path + "_L1Quality";
      //     elements_[l1Name.c_str()] = 
      //       dbe_->book1D("L1Quality", "Quality of L1 Muons", 8, 0, 8);
      //     for (size_t i = 0; i < 8; i++)
      //       elements_[l1Name.c_str()]->setBinLabel(i + 1, Form("%i", i));

      for (size_t i = 0; i < sources.size(); i++) {
        string source = sources[i];
        for (size_t j = 0; j < stepLabels_[path].size(); j++) {
          bookHist(path, stepLabels_[path][j], source, "Eta");
          bookHist(path, stepLabels_[path][j], source, "Phi");
          bookHist(path, stepLabels_[path][j], source, "MaxPt1");
          bookHist(path, stepLabels_[path][j], source, "MaxPt2");
        }
      }

    }

  }

}



void 
HLTMuonValidator::analyze(const Event & iEvent, const EventSetup & iSetup)
{

  static int eventNumber = 0;
  eventNumber++;
  LogTrace("HLTMuonVal") << "In HLTMuonValidator::analyze,  " 
                         << "Event: " << eventNumber;

  Handle<          TriggerEventWithRefs> rawTriggerEvent;
  Handle<                MuonCollection> recMuons;
  Handle<         GenParticleCollection> genParticles;

  iEvent.getByLabel("hltTriggerSummaryRAW", rawTriggerEvent);
  if (rawTriggerEvent.failedToGet())
    {LogError("HLTMuonVal") << "No trigger summary found"; return;}
  iEvent.getByLabel(    recMuonLabel_, recMuons     );
  iEvent.getByLabel(genParticleLabel_, genParticles );

  vector<string> sources;
  if (genParticles.isValid()) sources.push_back("gen");
  if (    recMuons.isValid()) sources.push_back("rec");

  for (size_t sourceNo = 0; sourceNo < sources.size(); sourceNo++) {

    string source = sources[sourceNo];
    
    // If this is the first event, initialize selectors
    if (!genMuonSelector_) genMuonSelector_ =
      new StringCutObjectSelector<reco::GenParticle>(genMuonCut_);
    if (!recMuonSelector_) recMuonSelector_ =
      new StringCutObjectSelector<reco::Muon       >(recMuonCut_);

    // Make each good gen/rec muon into the base cand for a MatchStruct
    vector<MatchStruct> matches;
    if (source == "gen" && genParticles.isValid())
      for (size_t i = 0; i < genParticles->size(); i++)
        if ((*genMuonSelector_)(genParticles->at(i)))
          matches.push_back(MatchStruct(& genParticles->at(i)));
    if (source == "rec" && recMuons.isValid())
      for (size_t i = 0; i < recMuons->size(); i++)
        if ((*recMuonSelector_)(recMuons->at(i)))
          matches.push_back(MatchStruct(& recMuons->at(i)));
    
    // Sort the MatchStructs by pT for later filling of turn-on curve
    sort(matches.begin(), matches.end(), matchesByDescendingPt());

    set<string>::iterator iPath;
    for (iPath = hltPaths_.begin(); iPath != hltPaths_.end(); iPath++)
      analyzePath(iEvent, * iPath, source, matches, rawTriggerEvent);

  } // End loop over sources

}



void 
HLTMuonValidator::analyzePath(const Event & iEvent,
                              const string & path, 
                              const string & source,
                              vector<MatchStruct> matches,
                              Handle<TriggerEventWithRefs> rawTriggerEvent)
{

  const bool skipFilters = (path == "NoFilters");

  const float maxEta = elements_[path + "_" + "CutMaxEta"]->getFloatValue();
  const bool isDoubleMuonPath = (path.find("Double") != string::npos);
  const size_t nFilters   = filterLabels_[path].size();
  const size_t nSteps     = stepLabels_[path].size();
  const size_t nStepsHlt  = nSteps - 2;
  const int nObjectsToPassPath = (isDoubleMuonPath) ? 2 : 1;
  vector< L1MuonParticleRef > candsL1;
  vector< vector< RecoChargedCandidateRef      > > refsHlt(nStepsHlt);
  vector< vector< const RecoChargedCandidate * > > candsHlt(nStepsHlt);

  for (size_t i = 0; i < nFilters; i++) {
    const int hltStep = i - 1;
    InputTag tag     = InputTag(filterLabels_[path][i], "", hltProcessName_);
    size_t   iFilter = rawTriggerEvent->filterIndex(tag);
    if (iFilter < rawTriggerEvent->size()) {
      if (i == 0) 
        rawTriggerEvent->getObjects(iFilter, TriggerL1Mu, candsL1);
      else
        rawTriggerEvent->getObjects(iFilter, TriggerMuon, 
                                    refsHlt[hltStep]);
    }
    else if (!skipFilters)
      LogTrace("HLTMuonVal") << "No collection with label " << tag;
  }
  if (skipFilters) {
      Handle<RecoChargedCandidateCollection> handleCandsL2;
      Handle<RecoChargedCandidateCollection> handleCandsL3;
      iEvent.getByLabel(l2CandLabel_, handleCandsL2);
      iEvent.getByLabel(l3CandLabel_, handleCandsL3);
      if (handleCandsL2.isValid())
        for (size_t i = 0; i < handleCandsL2->size(); i++)
          candsHlt[0].push_back(& handleCandsL2->at(i));
      if (handleCandsL3.isValid())
        for (size_t i = 0; i < handleCandsL3->size(); i++)
          candsHlt[1].push_back(& handleCandsL3->at(i));
  }
  else for (size_t i = 0; i < nStepsHlt; i++)
    for (size_t j = 0; j < refsHlt[i].size(); j++)
      candsHlt[i].push_back(& * refsHlt[i][j]);

  // Add trigger objects to the MatchStructs
  findMatches(matches, candsL1, candsHlt);

  vector<size_t> matchesInEtaRange;
  vector<bool> hasMatch(matches.size(), true);

  for (size_t step = 0; step < nSteps; step++) {

    const size_t hltStep = (step >= 2) ? step - 2 : 0;
    const size_t level   = (step == 1) ? 1 :
                           (step == 2) ? 2 :
                           (step == 3) ? ((nStepsHlt == 4) ? 2 : 3) :
                           (step >= 4) ? 3 :
                           0; // default value when step == 0

    for (size_t j = 0; j < matches.size(); j++) {
      if (level == 0) {
        if (fabs(matches[j].candBase->eta()) < maxEta)
          matchesInEtaRange.push_back(j);
      }
      else if (level == 1) {
        if (matches[j].candL1 == 0)
          hasMatch[j] = false;
      }
      else if (level >= 2) {
        if (matches[j].candHlt[hltStep] == 0)
          hasMatch[j] = false;
        else if (!hasMatch[j]) {
          LogTrace("HLTMuonVal") << "Match found for HLT step " << hltStep
                                 << " of " << nStepsHlt 
                                 << " without previous match!";
          break;
        }
      }
    }

    if (std::count(hasMatch.begin(), hasMatch.end(), true) <
        nObjectsToPassPath) 
      break;

    string pre  = path + "_" + source + "Pass";
    string post = "_" + stepLabels_[path][step];

    for (size_t j = 0; j < matches.size(); j++) {
      float pt  = matches[j].candBase->pt();
      float eta = matches[j].candBase->eta();
      float phi = matches[j].candBase->phi();
      if (hasMatch[j]) { 
        if (matchesInEtaRange.size() >= 1 && j == matchesInEtaRange[0])
          elements_[pre + "MaxPt1" + post]->Fill(pt);
        if (matchesInEtaRange.size() >= 2 && j == matchesInEtaRange[1])
          elements_[pre + "MaxPt2" + post]->Fill(pt);
        if(fabs(eta) < maxEta && pt > cutsMinPt_[path]) {
          elements_[pre + "Eta" + post]->Fill(eta);
          elements_[pre + "Phi" + post]->Fill(phi);
        }
      }
    }

  }

}



void
HLTMuonValidator::findMatches(
    vector<MatchStruct> & matches,
    vector<L1MuonParticleRef> candsL1,
    vector< vector< const RecoChargedCandidate *> > candsHlt)
{

  set<size_t>::iterator it;

  set<size_t> indicesL1;
  for (size_t i = 0; i < candsL1.size(); i++) 
    indicesL1.insert(i);

  vector< set<size_t> > indicesHlt(candsHlt.size());
  for (size_t i = 0; i < candsHlt.size(); i++)
    for (size_t j = 0; j < candsHlt[i].size(); j++)
      indicesHlt[i].insert(j);

  for (size_t i = 0; i < matches.size(); i++) {

    const Candidate * cand = matches[i].candBase;

    double bestDeltaR = cutsDr_[0];
    size_t bestMatch = kNull;
    for (it = indicesL1.begin(); it != indicesL1.end(); it++) {
      double dR = deltaR(cand->eta(), cand->phi(),
                         candsL1[*it]->eta(), candsL1[*it]->phi());
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
    }
    if (bestMatch != kNull)
      matches[i].candL1 = & * candsL1[bestMatch];
    indicesL1.erase(bestMatch);

    matches[i].candHlt.assign(candsHlt.size(), 0);
    for (size_t j = 0; j < candsHlt.size(); j++) {
      size_t level = (candsHlt.size() == 4) ? (j < 2) ? 2 : 3 :
                     (candsHlt.size() == 2) ? (j < 1) ? 2 : 3 :
                     2;
      bestDeltaR = cutsDr_[level - 2];
      bestMatch = kNull;
      for (it = indicesHlt[j].begin(); it != indicesHlt[j].end(); it++) {
        double dR = deltaR(cand->eta(), cand->phi(),
                           candsHlt[j][*it]->eta(), candsHlt[j][*it]->phi());
        if (dR < bestDeltaR) {
          bestMatch = *it;
          bestDeltaR = dR;
        }
      }
      if (bestMatch != kNull)
        matches[i].candHlt[j] = candsHlt[j][bestMatch];
      indicesHlt[j].erase(bestMatch);
    }

//     cout << "    Muon: " << cand->eta() << ", ";
//     if (matches[i].candL1) cout << matches[i].candL1->eta() << ", ";
//     else cout << "none, ";
//     for (size_t j = 0; j < candsHlt.size(); j++) 
//       if (matches[i].candHlt[j]) cout << matches[i].candHlt[j]->eta() << ", ";
//       else cout << "none, ";
//     cout << endl;

  }

}




void 
HLTMuonValidator::bookHist(string path, string label, 
                           string source, string type)
{

  string sourceUpper = source; 
  sourceUpper[0] = toupper(sourceUpper[0]);
  string name  = source + "Pass" + type + "_" + label;
  string rootName = path + "_" + name;
  TH1F * h;

  if (type.find("MaxPt") != string::npos) {
    string desc = (type == "MaxPt1") ? "Leading" : "Next-to-Leading";
    string title = "pT of " + desc + " " + sourceUpper + " Muon "+
                   "matched to " + label;
    const size_t nBins = parametersTurnOn_.size() - 1;
    float * edges = new float[nBins + 1];
    for (size_t i = 0; i < nBins + 1; i++) edges[i] = parametersTurnOn_[i];
    h = new TH1F(rootName.c_str(), title.c_str(), nBins, edges);
  }

  else {
    string symbol = (type == "Eta") ? "#eta" : "#phi";
    string title  = symbol + " of " + sourceUpper + " Muons " +
                    "matched to " + label;
    vector<double> params = (type == "Eta") ? parametersEta_ : parametersPhi_; 
    int    nBins = (int)params[0];
    double min   = params[1];
    double max   = params[2];
    h = new TH1F(rootName.c_str(), title.c_str(), nBins, min, max);
  }

  h->Sumw2();
  elements_[rootName] = dbe_->book1D(name, h);
  delete h;

}



DEFINE_FWK_MODULE(HLTMuonValidator);
