 /** \file HLTMuonValidator.cc
 *  $Date: 2009/10/20 23:12:08 $
 *  $Revision: 1.4 $
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
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

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

  cutMinPt_    = pset.getParameter< double         >("cutMinPt"   );
  cutMotherId_ = pset.getParameter< unsigned int   >("cutMotherId");
  cutsDr_      = pset.getParameter< vector<double> >("cutsDr"     );

  parametersEta_    = pset.getParameter< vector<double> >("parametersEta");
  parametersPhi_    = pset.getParameter< vector<double> >("parametersPhi");
  parametersTurnOn_ = pset.getParameter< vector<double> >("parametersTurnOn");

  dbe_ = Service<DQMStore>().operator->();
  dbe_->setVerbose(0);

}



void 
HLTMuonValidator::beginJob(const EventSetup & iSetup) 
{

  HLTConfigProvider hltConfig;
  hltConfig.init(hltProcessName_);
  vector<string> validTriggerNames = hltConfig.triggerNames();

  for (size_t i = 0; i < hltPathsToCheck_.size(); i++) {
    TPRegexp pattern(hltPathsToCheck_[i]);
    for (size_t j = 0; j < validTriggerNames.size(); j++)
      if (TString(validTriggerNames[j]).Contains(pattern))
        hltPaths_.insert(validTriggerNames[j]);
  }

  set<string>::iterator pathIter;

  for (pathIter = hltPaths_.begin(); pathIter != hltPaths_.end(); pathIter++) {
 
    string path = * pathIter;
    vector<string> moduleLabels = hltConfig.moduleLabels(path);

    for (size_t i = 0; i < moduleLabels.size(); i++)
      if (moduleLabels[i].find("Filtered") != string::npos)
        filterLabels_[path].push_back(moduleLabels[i]);

    dbe_->setCurrentFolder("HLT/Muon/Distributions/" + path);
    elements_[path + "_" + "CutMinPt" ] = dbe_->bookFloat("CutMinPt" );
    elements_[path + "_" + "CutMaxEta"] = dbe_->bookFloat("CutMaxEta");
    elements_[path + "_" + "CutMinPt" ]->Fill(cutMinPt_ );
    if (TString(path).Contains(kLooseL1Requirement))
      elements_[path + "_" + "CutMaxEta"]->Fill(2.4);
    else
      elements_[path + "_" + "CutMaxEta"]->Fill(2.1);

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

    for (size_t i = 0; i < 2; i++) {
      string source = kSources[i];
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
  LogTrace("HLTMuonVal") << "In HLTMuonValidator::analyze,  " 
                         << "Event: " << eventNumber++;

  Handle<TriggerEventWithRefs> rawTriggerEvent;
  Handle<                MuonCollection> recMuons;
  Handle<         GenParticleCollection> genParticles;
  Handle<      L1MuonParticleCollection> handleCandsL1;
  Handle<RecoChargedCandidateCollection> handleCandsL2;
  Handle<RecoChargedCandidateCollection> handleCandsL3;

  iEvent.getByLabel("hltTriggerSummaryRAW", rawTriggerEvent);
  if (rawTriggerEvent.failedToGet()) 
    {LogError("HLTMuonVal") << "No RAW trigger summary found"; return;}

  iEvent.getByLabel("muons",               genParticles );
  iEvent.getByLabel("genParticles",        genParticles );
  iEvent.getByLabel("hltL1extraParticles", handleCandsL1);
  iEvent.getByLabel("hltL2MuonCandidates", handleCandsL2);
  iEvent.getByLabel("hltL3MuonCandidates", handleCandsL3);
  
  L1MuonParticleCollection candsL1;
  if (!handleCandsL1.isValid()) // Check for FastSim L1 collection
    iEvent.getByLabel("l1ParamMuons", handleCandsL1);
  if (handleCandsL1.isValid()) candsL1 = * handleCandsL1;
  else {
    // L1 collections not properly saved; try the trigger summary
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

  for (size_t k = 0; k < 2; k++) {

    string source = kSources[k];

    vector<MatchStruct> matches;

    if (source == "gen" && genParticles.isValid())
      for (size_t i = 0; i < genParticles->size(); i++) {
        const GenParticle * genParticle = & genParticles->at(i);
        const Candidate * mother = findMother(genParticle);
        int    momId  = mother ? mother->pdgId() : 0;
        int    id     = genParticle->pdgId();
        int    status = genParticle->status();
        if (abs(id) == 13 && status == 1 && 
            (cutMotherId_ == 0 || abs(momId) == cutMotherId_))
          matches.push_back(MatchStruct(genParticle));
      }
    else if (source == "rec" && recMuons.isValid())
      for (size_t i = 0; i < recMuons->size(); i++) {
        const Muon * muon = & recMuons->at(i);
        if (muon->isGlobalMuon())
          matches.push_back(MatchStruct(muon));
      }
    sort(matches.begin(), matches.end(), matchesByDescendingPt());

    for (size_t i = 0; i < candsL3.size(); i++) {
      size_t match = findMatch(& candsL3[i], matches, cutsDr_[2], "L3");
      if (match == kNull) LogTrace("HLTMuonVal") << "Orphan L3 in " 
                                                 << kSources[k] << endl;
      else matches[match].setL3(& candsL3[i]);
    }

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
        LogTrace("HLTMuonVal") << "Orphan L2 in " << kSources[k] << endl;
      else matches[match].setL2(& candsL2[i]);
    }

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
        LogTrace("HLTMuonVal") << "Orphan L1 in " << kSources[k] << endl;
      else matches[match].setL1(& candsL1[i]);
    }

    set<string>::iterator pathIter;
    for (pathIter = hltPaths_.begin(); pathIter != hltPaths_.end(); pathIter++)
      analyzePath(* pathIter, kSources[k], matches, rawTriggerEvent);

  }

}



void 
HLTMuonValidator::analyzePath(const string & path, 
                              const string & source,
                              vector<MatchStruct> & matches,
                              Handle<TriggerEventWithRefs> & rawTriggerEvent)
{

  const float maxEta = elements_[path + "_" + "CutMaxEta" ]->getFloatValue();
  const bool isDoubleMuonPath = (path.find("Double") != string::npos);
  const size_t nFilters   = filterLabels_[path].size();
  const size_t nSteps     = stepLabels_[path].size();
  const size_t nStepsHlt  = nSteps - 2;
  const size_t nObjectsToPassPath = (isDoubleMuonPath) ? 2 : 1;
  vector< L1MuonParticleRef                 > candsPassingL1;
  vector< vector< RecoChargedCandidateRef > > candsPassingHlt(nStepsHlt);

  for (size_t i = 0; i < nFilters; i++) {
    InputTag tag     = InputTag(filterLabels_[path][i], "", hltProcessName_);
    size_t   iFilter = rawTriggerEvent->filterIndex(tag);
    if (iFilter < rawTriggerEvent->size())
      if (i == 0)
        rawTriggerEvent->getObjects(iFilter, TriggerL1Mu, candsPassingL1);
      else
        rawTriggerEvent->getObjects(iFilter, TriggerMuon, candsPassingHlt[i-1]);
    else LogTrace("HLTMuonVal") << "No collection with label " << tag;
  }

  vector<bool> missingMatch(nSteps, false);
  vector< vector<bool> > hasMatch(nSteps);
  hasMatch[0].assign(matches.size(), true);
  for (size_t i = 1; i < nSteps; i++)
    hasMatch[i].assign(matches.size(), false);
  vector<size_t> matchesInRange;

  for (size_t iStep = 0; iStep < nSteps; iStep++) {

    const size_t hltStep = (iStep >= 2) ? iStep - 2 : 0;
    const size_t level   = (iStep == 1) ? 1 :
                           (iStep == 2) ? 2 :
                           (iStep == 3) ? (nStepsHlt == 4) ? 2 : 3 :
                           (iStep >= 4) ? 3 :
                           0;

    size_t nMatches = 0;
    if (level == 0) {
      nMatches = matches.size();
      for (size_t j = 0; j < matches.size(); j++)
        if ((matches[j].candBase->eta()) < maxEta)
          matchesInRange.push_back(j);
    }
    if (level == 1)
      for (size_t j = 0; j < matches.size(); j++) 
        for (size_t k = 0; k < candsPassingL1.size(); k++) 
          if (identical(matches[j].candL1, & * candsPassingL1[k])) {
            hasMatch[iStep][j] = true;
            nMatches++;
          }
    if (level >= 2)
      for (size_t j = 0; j < matches.size(); j++)
        for (size_t k = 0; k < candsPassingHlt[hltStep].size(); k++)
          if (identical(matches[j].candHlt[level - 2],
                        & * candsPassingHlt[hltStep][k])) {
            hasMatch[iStep][j] = true;
            nMatches++;
          }

    if (nMatches < nObjectsToPassPath) break;

    string pre  = path + "_" + source + "Pass";
    string post = "_" + stepLabels_[path][iStep];

    for (size_t j = 0; j < matches.size(); j++) {
      if (!hasMatch[iStep][j]) missingMatch[j] = true;
      double pt  = matches[j].candBase->pt();
      double eta = matches[j].candBase->eta();
      double phi = matches[j].candBase->phi();
      if (!missingMatch[j]) { 
        if (matchesInRange.size() >= 1 && j == matchesInRange[0])
          elements_[pre + "MaxPt1" + post]->Fill(pt);
        if (matchesInRange.size() >= 2 && j == matchesInRange[1])
          elements_[pre + "MaxPt2" + post]->Fill(pt);
        if(fabs(eta) < maxEta && pt > cutMinPt_) {
          elements_[pre + "Eta" + post]->Fill(eta);
          elements_[pre + "Phi" + post]->Fill(phi);
        }
      }
    }

  }

}



const reco::Candidate * 
HLTMuonValidator::findMother(const reco::Candidate* p) 
{
  const reco::Candidate* mother = p->mother();
  if (mother) {
    if (mother->pdgId() == p->pdgId()) return findMother(mother);
    else return mother;
  }
  else return 0;
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
