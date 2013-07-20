
/** \file HLTMuonPlotter.cc
 *  $Date: 2013/04/19 23:22:27 $
 *  $Revision: 1.2 $
 */



#include "HLTriggerOffline/Muon/interface/HLTMuonPlotter.h"
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



HLTMuonPlotter::HLTMuonPlotter(const ParameterSet & pset,
                               string hltPath,
                               const std::vector<string>& moduleLabels,
                               const std::vector<string>& stepLabels) :
  l1Matcher_(pset)
{

  hltPath_ = hltPath;
  moduleLabels_ = moduleLabels;
  stepLabels_ = stepLabels;
  hltProcessName_  = pset.getParameter<string>("hltProcessName");

  genParticleLabel_ = pset.getParameter<string>("genParticleLabel");
      recMuonLabel_ = pset.getParameter<string>(    "recMuonLabel");

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
HLTMuonPlotter::beginJob() 
{
}



void 
HLTMuonPlotter::beginRun(const Run & iRun, const EventSetup & iSetup) 
{

  static int runNumber = 0;
  runNumber++;

  l1Matcher_.init(iSetup);

  cutMaxEta_ = 2.4;
  if (hltPath_.find("eta2p1") != string::npos) 
    cutMaxEta_ = 2.1;

  // Choose a pT cut for gen/rec muons based on the pT cut in the hltPath_
  unsigned int threshold = 0;
  TPRegexp ptRegexp("Mu([0-9]+)");
  TObjArray * regexArray = ptRegexp.MatchS(hltPath_);
  if (regexArray->GetEntriesFast() == 2) {
    threshold = atoi(((TObjString *)regexArray->At(1))->GetString());
  }
  delete regexArray;
  // We select a whole number min pT cut slightly above the hltPath_'s final 
  // pt threshold, then subtract a bit to let through particle gun muons with
  // exact integer pT:
  cutMinPt_ = ceil(threshold * 1.1) - 0.01;
  if (cutMinPt_ < 0.) cutMinPt_ = 0.;
  
  string baseDir = "HLT/Muon/Distributions/";
  dbe_->setCurrentFolder(baseDir + hltPath_);

  vector<string> sources(2);
  sources[0] = "gen";
  sources[1] = "rec";

  if (dbe_->get(baseDir + hltPath_ + "/CutMinPt") == 0) {

      elements_["CutMinPt" ] = dbe_->bookFloat("CutMinPt" );
      elements_["CutMaxEta"] = dbe_->bookFloat("CutMaxEta");
      elements_["CutMinPt" ]->Fill(cutMinPt_);
      elements_["CutMaxEta"]->Fill(cutMaxEta_);

      for (size_t i = 0; i < sources.size(); i++) {
        string source = sources[i];
        for (size_t j = 0; j < stepLabels_.size(); j++) {
          bookHist(hltPath_, stepLabels_[j], source, "Eta");
          bookHist(hltPath_, stepLabels_[j], source, "Phi");
          bookHist(hltPath_, stepLabels_[j], source, "MaxPt1");
          bookHist(hltPath_, stepLabels_[j], source, "MaxPt2");
        }
      }
  }

}



void 
HLTMuonPlotter::analyze(const Event & iEvent, const EventSetup & iSetup)
{

  static int eventNumber = 0;
  eventNumber++;
  LogTrace("HLTMuonVal") << "In HLTMuonPlotter::analyze,  " 
                         << "Event: " << eventNumber;

  // cout << hltPath_ << endl;
  // for (size_t i = 0; i < moduleLabels_.size(); i++)
  //   cout << "    " << moduleLabels_[i] << endl;

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

    const bool isDoubleMuonPath = (hltPath_.find("Double") != string::npos);
    const size_t nFilters   = moduleLabels_.size();
    const size_t nSteps     = stepLabels_.size();
    const size_t nStepsHlt  = nSteps - 2;
    const int nObjectsToPassPath = (isDoubleMuonPath) ? 2 : 1;
    vector< L1MuonParticleRef > candsL1;
    vector< vector< RecoChargedCandidateRef      > > refsHlt(nStepsHlt);
    vector< vector< const RecoChargedCandidate * > > candsHlt(nStepsHlt);
    
    for (size_t i = 0; i < nFilters; i++) {
      const int hltStep = i - 1;
      InputTag tag = InputTag(moduleLabels_[i], "", hltProcessName_);
      size_t iFilter = rawTriggerEvent->filterIndex(tag);
      if (iFilter < rawTriggerEvent->size()) {
        if (i == 0)
          rawTriggerEvent->getObjects(iFilter, TriggerL1Mu, candsL1);
        else
          rawTriggerEvent->getObjects(iFilter, TriggerMuon, 
                                      refsHlt[hltStep]);
      }
      else LogTrace("HLTMuonVal") << "No collection with label " << tag;
    }
    for (size_t i = 0; i < nStepsHlt; i++)
      for (size_t j = 0; j < refsHlt[i].size(); j++)
        if (refsHlt[i][j].isAvailable()) {
          candsHlt[i].push_back(& * refsHlt[i][j]);
        } else {
          LogWarning("HLTMuonPlotter")
            << "Ref refsHlt[i][j]: product not available "
            << i << " " << j;
        }
    
    // Add trigger objects to the MatchStructs
    findMatches(matches, candsL1, candsHlt);
    
    vector<size_t> matchesInEtaRange;
    vector<bool> hasMatch(matches.size(), true);
    
    for (size_t step = 0; step < nSteps; step++) {
      
      const size_t hltStep = (step >= 2) ? step - 2 : 0;
      size_t level = 0;
      if (stepLabels_[step].find("L3") != string::npos) level = 3;
      else if (stepLabels_[step].find("L2") != string::npos) level = 2;
      else if (stepLabels_[step].find("L1") != string::npos) level = 1;
      
      for (size_t j = 0; j < matches.size(); j++) {
        if (level == 0) {
          if (fabs(matches[j].candBase->eta()) < cutMaxEta_)
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
      
      string pre  = source + "Pass";
      string post = "_" + stepLabels_[step];
      
      for (size_t j = 0; j < matches.size(); j++) {
        float pt  = matches[j].candBase->pt();
        float eta = matches[j].candBase->eta();
        float phi = matches[j].candBase->phi();
        if (hasMatch[j]) { 
          if (matchesInEtaRange.size() >= 1 && j == matchesInEtaRange[0])
            elements_[pre + "MaxPt1" + post]->Fill(pt);
          if (matchesInEtaRange.size() >= 2 && j == matchesInEtaRange[1])
            elements_[pre + "MaxPt2" + post]->Fill(pt);
          if (pt > cutMinPt_) {
            elements_[pre + "Eta" + post]->Fill(eta);
            if (fabs(eta) < cutMaxEta_)
              elements_[pre + "Phi" + post]->Fill(phi);
          }
        }
      }
      
    }
    
    
    
  } // End loop over sources
  
}



void
HLTMuonPlotter::findMatches(
    vector<MatchStruct> & matches,
    const std::vector<L1MuonParticleRef>& candsL1,
    const std::vector< vector< const RecoChargedCandidate *> >& candsHlt)
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
     if (candsL1[*it].isAvailable()) {
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
     } else {
       LogWarning("HLTMuonPlotter")
	 << "Ref candsL1[*it]: product not available "
	 << *it;
     }
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
HLTMuonPlotter::bookHist(string path, string label, 
                         string source, string type)
{

  string sourceUpper = source; 
  sourceUpper[0] = toupper(sourceUpper[0]);
  string name = source + "Pass" + type + "_" + label;
  TH1F * h;

  if (type.find("MaxPt") != string::npos) {
    string desc = (type == "MaxPt1") ? "Leading" : "Next-to-Leading";
    string title = "pT of " + desc + " " + sourceUpper + " Muon "+
                   "matched to " + label;
    const size_t nBins = parametersTurnOn_.size() - 1;
    float * edges = new float[nBins + 1];
    for (size_t i = 0; i < nBins + 1; i++) edges[i] = parametersTurnOn_[i];
    h = new TH1F(name.c_str(), title.c_str(), nBins, edges);
  }

  else {
    string symbol = (type == "Eta") ? "#eta" : "#phi";
    string title  = symbol + " of " + sourceUpper + " Muons " +
                    "matched to " + label;
    vector<double> params = (type == "Eta") ? parametersEta_ : parametersPhi_; 
    int    nBins = (int)params[0];
    double min   = params[1];
    double max   = params[2];
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  }

  h->Sumw2();
  elements_[name] = dbe_->book1D(name, h);
  delete h;

}

