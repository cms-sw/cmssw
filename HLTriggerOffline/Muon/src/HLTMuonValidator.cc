/** \file HLTMuonValidator.cc
 *  $Date: 2010/02/14 19:19:48 $
 *  $Revision: 1.16 $
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



HLTMuonValidator::HLTMuonValidator(const ParameterSet & pset)
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

  bool changedConfig;
  if (!hltConfig_.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    LogError("Initialization of HLTConfigProvider failed!!"); 
    return;
  }
  if (runNumber > 1 && changedConfig)
    LogWarning("The HLT configuration changed... results might get funky.");

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

}



void
HLTMuonValidator::initializeHists(vector<string> sources)
{

  vector<string> validTriggerNames = hltConfig_.triggerNames();

  set<string>::iterator iPath;
  TPRegexp suffixPtCut("[0-9]+$");

  for (iPath = hltPaths_.begin(); iPath != hltPaths_.end(); iPath++) {
 
    string path = * iPath;
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

    dbe_->setCurrentFolder("HLT/Muon/Distributions/" + path);
    elements_[path + "_" + "CutMinPt" ] = dbe_->bookFloat("CutMinPt" );
    elements_[path + "_" + "CutMaxEta"] = dbe_->bookFloat("CutMaxEta");
    elements_[path + "_" + "CutMinPt" ]->Fill(cutMinPt);
    elements_[path + "_" + "CutMaxEta"]->Fill(cutMaxEta);

    if (path == "NoFilters") {
      filterLabels_[path].push_back("hltL1sL1SingleMuOpenL1SingleMu0");
      // The L2 and L3 collections will be built manually later on
      filterLabels_[path].push_back("");
      filterLabels_[path].push_back("");
    }

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

    string l1Name = path + "_L1Quality";
    elements_[l1Name.c_str()] = 
      dbe_->book1D("L1Quality", "Quality of L1 Muons", 8, 0, 8);
    for (size_t i = 0; i < 8; i++)
      elements_[l1Name.c_str()]->setBinLabel(i + 1, Form("%i", i));

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
  Handle<      L1MuonParticleCollection> handleCandsL1;
  Handle<RecoChargedCandidateCollection> handleCandsL2;
  Handle<RecoChargedCandidateCollection> handleCandsL3;

  iEvent.getByLabel("hltTriggerSummaryRAW", rawTriggerEvent);
  if (rawTriggerEvent.failedToGet())
    {LogError("HLTMuonVal") << "No trigger summary found"; return;}

  iEvent.getByLabel(    recMuonLabel_, recMuons     );
  iEvent.getByLabel(genParticleLabel_, genParticles );
  iEvent.getByLabel(     l1CandLabel_, handleCandsL1);
  iEvent.getByLabel(     l2CandLabel_, handleCandsL2);
  iEvent.getByLabel(     l3CandLabel_, handleCandsL3);

  vector<string> sources;
  if (genParticles.isValid()) sources.push_back("gen");
  if (    recMuons.isValid()) sources.push_back("rec");

  bool isFirstEvent = (eventNumber == 1);
  if (isFirstEvent) initializeHists(sources);

  L1MuonParticleCollection candsL1;
  if (handleCandsL1.isValid()) 
    candsL1 = * handleCandsL1;
  // If the L1 collection was not properly saved, let's get 
  // complicated and extract it from the trigger summary
  else { 
    InputTag tag = InputTag("hltL1MuOpenL1Filtered0", "", hltProcessName_);
    size_t iFilter = rawTriggerEvent->filterIndex(tag);
    vector<L1MuonParticleRef> candsPassingL1;
    if (iFilter < rawTriggerEvent->size())
      rawTriggerEvent->getObjects(iFilter, TriggerL1Mu, candsPassingL1);
    for (size_t i = 0; i < candsPassingL1.size(); i++) 
      candsL1.push_back(* candsPassingL1[i]);
  }

  vector<RecoChargedCandidateCollection> candsHlt(2);
  if (handleCandsL2.isValid()) candsHlt[0] = * handleCandsL2;
  if (handleCandsL3.isValid()) candsHlt[1] = * handleCandsL3;
  RecoChargedCandidateCollection & candsL2 = candsHlt[0];
  RecoChargedCandidateCollection & candsL3 = candsHlt[1];

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

    // Below, we match gen/reco to trigger, using the full collections of
    // trigger objects, so matching is consisten between paths.

    // Try to match L3's to gen/rec cands
    for (size_t i = 0; i < candsL3.size(); i++) {
      size_t match = findMatch(& candsL3[i], matches, cutsDr_[2], "L3");
      if (match == kNull) LogTrace("HLTMuonVal") << "Orphan L3 in " 
                                                 << source << endl;
      else matches[match].setL3(& candsL3[i]);
    }

    // Fill L2 matches by following seeds back from L3,
    // then try to match any remaining L2's
    for (size_t i = 0; i < candsL2.size(); i++) {
      size_t match = kNull;
      for (size_t j = 0; j < matches.size(); j++)
        if (matches[j].candL3() &&
            candsL2[i].track() == 
            matches[j].candL3()->track()->seedRef().
            castTo<L3MuonTrajectorySeedRef>()->l2Track())
          match = j;
      if (match == kNull) 
        match = findMatch(& candsL2[i], matches, cutsDr_[1], "L2");
      if (match == kNull) 
        LogTrace("HLTMuonVal") << "Orphan L2 in " << source << endl;
      else matches[match].setL2(& candsL2[i]);
    }

    // Fill L1 matches by following seeds back from L2,
    // then try to match any remaining L1's
    for (size_t i = 0; i < candsL1.size(); i++) {
      size_t match = kNull;
      for (size_t j = 0; j < matches.size(); j++)
        if (matches[j].candL2() && 
            identical(& candsL1[i], 
                      & * matches[j].candL2()->
                      track()->seedRef().
                      castTo< Ref<L2MuonTrajectorySeedCollection> >()->
                      l1Particle()))
          match = j;
      if (match == kNull) 
        match = findMatch(& candsL1[i], matches, cutsDr_[0], "L1");
      if (match == kNull) 
        LogTrace("HLTMuonVal") << "Orphan L1 in " << source << endl;
      else matches[match].setL1(& candsL1[i]);
    }

    set<string>::iterator iPath;
    for (iPath = hltPaths_.begin(); iPath != hltPaths_.end(); iPath++)
      analyzePath(* iPath, source, matches, rawTriggerEvent);

  } // End loop over sources

}



void 
HLTMuonValidator::analyzePath(const string & path, 
                              const string & source,
                              const vector<MatchStruct> & matches,
                              Handle<TriggerEventWithRefs> rawTriggerEvent)
{

  const bool skipFilters = (path == "NoFilters");

  const float maxEta = elements_[path + "_" + "CutMaxEta" ]->getFloatValue();
  const bool isDoubleMuonPath = (path.find("Double") != string::npos);
  const size_t nFilters   = filterLabels_[path].size();
  const size_t nSteps     = stepLabels_[path].size();
  const size_t nStepsHlt  = nSteps - 2;
  const size_t nObjectsToPassPath = (isDoubleMuonPath) ? 2 : 1;
  vector< L1MuonParticleRef > candsPassingL1;
  vector< vector< RecoChargedCandidateRef      > > refsPassingHlt(nStepsHlt);
  vector< vector< const RecoChargedCandidate * > > candsPassingHlt(nStepsHlt);

  // Get the collections of trigger objects that 
  // passed the filters in this particular path
  for (size_t i = 0; i < nFilters; i++) {
    const int hltStep = i - 1;
    InputTag tag     = InputTag(filterLabels_[path][i], "", hltProcessName_);
    size_t   iFilter = rawTriggerEvent->filterIndex(tag);
    if (iFilter < rawTriggerEvent->size()) {
      if (i == 0) 
        rawTriggerEvent->getObjects(iFilter, TriggerL1Mu, candsPassingL1);
      else
        rawTriggerEvent->getObjects(iFilter, TriggerMuon, 
                                    refsPassingHlt[hltStep]);
    }
    else if (!skipFilters)
      LogTrace("HLTMuonVal") << "No collection with label " << tag;
  }
  if (skipFilters) {
    for (size_t i = 0; i < matches.size(); i++)
      for (size_t j = 0; j < nStepsHlt; j++)
        if (matches[i].candHlt[j]) 
          candsPassingHlt[j].push_back(matches[i].candHlt[j]);
  }
  else for (size_t i = 0; i < nStepsHlt; i++)
    for (size_t j = 0; j < refsPassingHlt[i].size(); j++)
      candsPassingHlt[i].push_back(& * refsPassingHlt[i][j]);

  vector< vector<bool> > hasMatch(nSteps, 
                                  vector<bool>(matches.size(), false));
  const size_t initialStep = 0;
  hasMatch[initialStep].assign(matches.size(), true);

  vector<size_t> matchesInEtaRange;

  for (size_t step = 0; step < nSteps; step++) {

    const size_t hltStep = (step >= 2) ? step - 2 : 0;
    const size_t level   = (step == 1) ? 1 :
                           (step == 2) ? 2 :
                           (step == 3) ? ((nStepsHlt == 4) ? 2 : 3) :
                           (step >= 4) ? 3 :
                           0; // default value when step == 0

    for (size_t j = 0; j < matches.size(); j++)
      if (level == 0) {
        if (fabs(matches[j].candBase->eta()) < maxEta)
          matchesInEtaRange.push_back(j);
      }
      else if (level == 1) {
        for (size_t k = 0; k < candsPassingL1.size(); k++) 
          if (identical(matches[j].candL1, & * candsPassingL1[k])) {
            hasMatch[step][j] = true;
            int l1Quality = matches[j].candL1->gmtMuonCand().quality();
            elements_[path + "_L1Quality"]->Fill(l1Quality);
          }
      }
      else if (level >= 2) {
        for (size_t k = 0; k < candsPassingHlt[hltStep].size(); k++)
          if (identical(matches[j].candHlt[level - 2],
                        candsPassingHlt[hltStep][k]))
            hasMatch[step][j] = true;
      }
    
    size_t nMatches = count(hasMatch[step].begin(), 
                            hasMatch[step].end(), 
                            true);
    if (nMatches < nObjectsToPassPath) break;

    string pre  = path + "_" + source + "Pass";
    string post = "_" + stepLabels_[path][step];

    for (size_t j = 0; j < matches.size(); j++) {
      double pt  = matches[j].candBase->pt();
      double eta = matches[j].candBase->eta();
      double phi = matches[j].candBase->phi();
      if (hasMatch[step][j]) { 
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



bool
HLTMuonValidator::identical(const Candidate * p1, const Candidate * p2)
{
  if (p1 != 0 &&
      p2 != 0 &&
      p1->eta() == p2->eta() &&
      p1->phi() == p2->phi() &&
      p1->pt () == p2->pt ())
    return true;
  return false;
}



// Returns the index of the MatchStruct whose gen/reco muon gives the best
// deltaR match to cand; if none pass the cut, returns null value
unsigned int 
HLTMuonValidator::findMatch(const Candidate * cand, 
                            vector<MatchStruct> & matches,
                            double maxDeltaR, 
                            string level)
{
  const double eta = cand->eta();
  const double phi = cand->phi();
  double bestDeltaR = maxDeltaR;
  unsigned int bestMatch = kNull;
  for (size_t i = 0; i < matches.size(); i++) {
    if (level == "L3" && matches[i].candL3()) continue;
    if (level == "L2" && matches[i].candL2()) continue;
    if (level == "L1" && matches[i].candL1  ) continue;
    double dR = deltaR(eta, phi, 
                       matches[i].candBase->eta(), 
                       matches[i].candBase->phi());
    if (dR < bestDeltaR) {
      bestMatch  =  i;
      bestDeltaR = dR;
    }
  }
  return bestMatch;
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
