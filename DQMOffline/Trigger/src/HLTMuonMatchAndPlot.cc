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
#include "DataFormats/Candidate/interface/CandMatchMap.h"

#include <iostream>

//////////////////////////////////////////////////////////////////////////////
//////// Namespaces and Typedefs /////////////////////////////////////////////

using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;
using namespace l1extra;

typedef std::vector<std::string> vstring;


//////////////////////////////////////////////////////////////////////////////
//////// HLTMuonMatchAndPlot Class Members ///////////////////////////////////

/// Constructor
HLTMuonMatchAndPlot::HLTMuonMatchAndPlot(const ParameterSet & pset, 
                                         string hltPath, 
                                         const vector<string>& moduleLabels) :
  hltProcessName_(pset.getParameter<string>("hltProcessName")),
  destination_(pset.getUntrackedParameter<string>("destination")),
  requiredTriggers_(pset.getUntrackedParameter<vstring>("requiredTriggers")),
  targetParams_(pset.getParameterSet("targetParams")),
  probeParams_(pset.getParameterSet("probeParams")),
  hltPath_(hltPath),
  moduleLabels_(moduleLabels),
  hasTargetRecoCuts(targetParams_.exists("recoCuts")),
  hasProbeRecoCuts(probeParams_.exists("recoCuts")),
  targetMuonSelector_(targetParams_.getUntrackedParameter<string>("recoCuts", "")),
  targetZ0Cut_(targetParams_.getUntrackedParameter<double>("z0Cut",0.)),
  targetD0Cut_(targetParams_.getUntrackedParameter<double>("d0Cut",0.)),
  probeMuonSelector_(probeParams_.getUntrackedParameter<string>("recoCuts", "")),
  probeZ0Cut_(probeParams_.getUntrackedParameter<double>("z0Cut",0.)),
  probeD0Cut_(probeParams_.getUntrackedParameter<double>("d0Cut",0.))
{

  // Create std::map<string, T> from ParameterSets. 
  fillMapFromPSet(binParams_, pset, "binParams");
  fillMapFromPSet(plotCuts_, pset, "plotCuts");

  // Prepare the DQMStore object.
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setVerbose(0);

  // Get the trigger level.
  triggerLevel_ = "L3";
  TPRegexp levelRegexp("L[1-3]");
  size_t nModules = moduleLabels_.size();
  TObjArray * levelArray = levelRegexp.MatchS(moduleLabels_[nModules - 1]);
  if (levelArray->GetEntriesFast() > 0) {
    triggerLevel_ = ((TObjString *)levelArray->At(0))->GetString();
  }
  delete levelArray;

  // Get the pT cut by parsing the name of the HLT path.
  cutMinPt_ = 3;
  TPRegexp ptRegexp("Mu([0-9]*)");
  TObjArray * objArray = ptRegexp.MatchS(hltPath_);
  if (objArray->GetEntriesFast() >= 2) {
    TObjString * ptCutString = (TObjString *)objArray->At(1);
    cutMinPt_ = atoi(ptCutString->GetString());
    cutMinPt_ = ceil(cutMinPt_ * plotCuts_["minPtFactor"]);
  }
  delete objArray;

}

void HLTMuonMatchAndPlot::beginRun(const edm::Run& iRun, 
                                   const edm::EventSetup& iSetup)
{

  TPRegexp suffixPtCut("Mu[0-9]+$");

  string baseDir = destination_;
  if (baseDir[baseDir.size() - 1] != '/') baseDir += '/';
  string pathSansSuffix = hltPath_;
  if (hltPath_.rfind("_v") < hltPath_.length())
    pathSansSuffix = hltPath_.substr(0, hltPath_.rfind("_v"));
  dbe_->setCurrentFolder(baseDir + pathSansSuffix);

  // Form is book1D(name, binningType, title) where 'binningType' is used 
  // to fetch the bin settings from binParams_.
  book1D("deltaR", "deltaR", ";#Deltar(reco, HLT);");
  book1D("hltPt", "pt", ";p_{T} of HLT object");
  book1D("hltEta", "eta", ";#eta of HLT object");
  book1D("hltPhi", "phi", ";#phi of HLT object");
  book1D("resolutionEta", "resolutionEta", ";#eta^{reco}-#eta^{HLT};");
  book1D("resolutionPhi", "resolutionPhi", ";#phi^{reco}-#phi^{HLT};");
  book1D("resolutionPt", "resolutionRel", 
         ";(p_{T}^{reco}-p_{T}^{HLT})/|p_{T}^{reco}|;");

  for (size_t i = 0; i < 2; i++) {

    string suffix = EFFICIENCY_SUFFIXES[i];

    book1D("efficiencyEta_" + suffix, "eta", ";#eta;");
    book1D("efficiencyPhi_" + suffix, "phi", ";#phi;");
    book1D("efficiencyTurnOn_" + suffix, "pt", ";p_{T};");
    book1D("efficiencyD0_" + suffix, "d0", ";d0;");
    book1D("efficiencyZ0_" + suffix, "z0", ";z0;");
    book1D("efficiencyCharge_" + suffix, "charge", ";charge;");
    book1D("efficiencyVertex_" + suffix, "NVertex", ";NVertex;");

    book2D("efficiencyPhiVsEta_" + suffix, "etaCoarse", "phiCoarse", 
           ";#eta;#phi");

    book1D("fakerateEta_" + suffix, "eta", ";#eta;");
    book1D("fakerateVertex_" + suffix, "NVertex", ";NVertex;");
    book1D("fakeratePhi_" + suffix, "phi", ";#phi;");
    book1D("fakerateTurnOn_" + suffix, "pt", ";p_{T};");

    book1D("massVsEtaZ_" + suffix, "etaCoarse", ";#eta");
    book1D("massVsEtaJpsi_" + suffix, "etaCoarse", ";#eta");
    book1D("massVsPtZ_" + suffix, "ptCoarse", ";p_{T}");
    book1D("massVsPtJpsi_" + suffix, "ptCoarse", ";p_{T}");
    book1D("massVsVertexZ_" + suffix, "NVertex", ";NVertex");
    book1D("massVsVertexJpsi_" + suffix, "NVertex", ";NVertex");

  }
  
}



void HLTMuonMatchAndPlot::endRun(const edm::Run& iRun, 
                                 const edm::EventSetup& iSetup)
{

}



void HLTMuonMatchAndPlot::analyze(Handle<MuonCollection>   & allMuons, 
				  Handle<BeamSpot>         & beamSpot,
				  Handle<VertexCollection> & vertices,
				  Handle<TriggerEvent>     & triggerSummary,  
				  Handle<TriggerResults>   & triggerResults)
{

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



  // Throw out this event if it doesn't pass the required triggers.
  for (size_t i = 0; i < requiredTriggers_.size(); i++) {
    unsigned int triggerIndex = triggerResults->find(requiredTriggers_[i]);
    if (triggerIndex < triggerResults->size() ||
        !triggerResults->accept(triggerIndex))
      return;
  }

  // Select objects based on the configuration.
  MuonCollection targetMuons = selectedMuons(* allMuons, * beamSpot, hasTargetRecoCuts, targetMuonSelector_, targetD0Cut_, targetZ0Cut_);
  MuonCollection probeMuons = selectedMuons(* allMuons, * beamSpot, hasProbeRecoCuts, probeMuonSelector_, probeD0Cut_, probeZ0Cut_);
  TriggerObjectCollection allTriggerObjects = triggerSummary->getObjects();
  TriggerObjectCollection hltMuons = 
    selectedTriggerObjects(allTriggerObjects, * triggerSummary, targetParams_);

  // Fill plots for HLT muons.
  for (size_t i = 0; i < hltMuons.size(); i++) {
    hists_["hltPt"]->Fill(hltMuons[i].pt());
    hists_["hltEta"]->Fill(hltMuons[i].eta());
    hists_["hltPhi"]->Fill(hltMuons[i].phi());
  }

  // Find the best trigger object matches for the targetMuons.
  vector<size_t> matches = matchByDeltaR(targetMuons, hltMuons, 
                                         plotCuts_[triggerLevel_ + "DeltaR"]);


  // Fill plots for matched muons.
  bool pairalreadyconsidered = false;
  for (size_t i = 0; i < targetMuons.size(); i++) {

    Muon & muon = targetMuons[i];

    // Fill plots which are not efficiencies.
    if (matches[i] < targetMuons.size()) {
      TriggerObject & hltMuon = hltMuons[matches[i]];
      double ptRes = (muon.pt() - hltMuon.pt()) / muon.pt();
      double etaRes = muon.eta() - hltMuon.eta();
      double phiRes = muon.phi() - hltMuon.phi();
      hists_["resolutionEta"]->Fill(etaRes);
      hists_["resolutionPhi"]->Fill(phiRes);
      hists_["resolutionPt"]->Fill(ptRes);
      hists_["deltaR"]->Fill(deltaR(muon, hltMuon));
    }

    // Fill numerators and denominators for efficiency plots.
    for (size_t j = 0; j < 2; j++) {

      string suffix = EFFICIENCY_SUFFIXES[j];

      // If no match was found, then the numerator plots don't get filled.
      if (suffix == "numer" && matches[i] >= targetMuons.size()) continue;

      if (muon.pt() > cutMinPt_) {
        hists_["efficiencyEta_" + suffix]->Fill(muon.eta());
        hists_["efficiencyPhiVsEta_" + suffix]->Fill(muon.eta(), muon.phi());
      }
      
      if (fabs(muon.eta()) < plotCuts_["maxEta"]) {
        hists_["efficiencyTurnOn_" + suffix]->Fill(muon.pt());
      }
      
      if (muon.pt() > cutMinPt_ && fabs(muon.eta()) < plotCuts_["maxEta"]) {
        const Track * track = 0;
        if (muon.isTrackerMuon()) track = & * muon.innerTrack();
        else if (muon.isStandAloneMuon()) track = & * muon.outerTrack();
        if (track) {
          double d0 = track->dxy(beamSpot->position());
          double z0 = track->dz(beamSpot->position());
          hists_["efficiencyVertex_" + suffix]->Fill(vertices->size());
          hists_["efficiencyPhi_" + suffix]->Fill(muon.phi());
          hists_["efficiencyD0_" + suffix]->Fill(d0);
          hists_["efficiencyZ0_" + suffix]->Fill(z0);
          hists_["efficiencyCharge_" + suffix]->Fill(muon.charge());
        }
      }
    }
      
    // Fill plots for tag and probe
    // Muon cannot be a tag because doesn't match an hlt muon     
    if(matches[i] >= targetMuons.size()) continue;
    for (size_t k = 0; k < targetMuons.size(); k++) {
      if(k == i) continue;
      if(muon.pt() < 20.0) continue;
      Muon & theProbe = targetMuons[k];
      if (muon.charge() != theProbe.charge() && !pairalreadyconsidered) {
        double mass = (muon.p4() + theProbe.p4()).M();
        if(mass > 60 && mass < 120) {
          hists_["massVsEtaZ_denom"]->Fill(theProbe.eta());
          hists_["massVsPtZ_denom"]->Fill(theProbe.pt());
          hists_["massVsVertexZ_denom"]->Fill(vertices->size());
          if(matches[k] < targetMuons.size()) {
            hists_["massVsEtaZ_numer"]->Fill(theProbe.eta());
            hists_["massVsPtZ_numer"]->Fill(theProbe.pt());
            hists_["massVsVertexZ_numer"]->Fill(vertices->size());
          }  
          pairalreadyconsidered = true;
        }
        if(mass > 1 && mass < 4) {
          hists_["massVsEtaJpsi_denom"]->Fill(theProbe.eta());
          hists_["massVsPtJpsi_denom"]->Fill(theProbe.pt());
          hists_["massVsVertexJpsi_denom"]->Fill(vertices->size());
          if(matches[k] < targetMuons.size()) {
            hists_["massVsEtaJpsi_numer"]->Fill(theProbe.eta());
            hists_["massVsPtJpsi_numer"]->Fill(theProbe.pt());
            hists_["massVsVertexJpsi_numer"]->Fill(vertices->size());
          }
          pairalreadyconsidered = true;
        }
      }
    } // End loop over denominator and numerator.
  } // End loop over targetMuons.

  // Plot fake rates (efficiency for HLT objects to not get matched to RECO).
  vector<size_t> hltMatches = matchByDeltaR(hltMuons, targetMuons,
                                            plotCuts_[triggerLevel_ + "DeltaR"]);
  for (size_t i = 0; i < hltMuons.size(); i++) {
    TriggerObject & hltMuon = hltMuons[i];
    bool isFake = hltMatches[i] > hltMuons.size();
    for (size_t j = 0; j < 2; j++) {
      string suffix = EFFICIENCY_SUFFIXES[j];
      // If match is found, then numerator plots should not get filled
      if (suffix == "numer" && ! isFake) continue;
      hists_["fakerateVertex_" + suffix]->Fill(vertices->size());
      hists_["fakerateEta_" + suffix]->Fill(hltMuon.eta());
      hists_["fakeratePhi_" + suffix]->Fill(hltMuon.phi());
      hists_["fakerateTurnOn_" + suffix]->Fill(hltMuon.pt());
    } // End loop over numerator and denominator.
  } // End loop over hltMuons.
  

} // End analyze() method.



// Method to fill binning parameters from a vector of doubles.
void 
HLTMuonMatchAndPlot::fillEdges(size_t & nBins, float * & edges, 
                               const vector<double>& binning) 
{

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
    for (size_t i = 0; i <= nBins; i++) edges[i] = min + (binwidth * i);
  } 

  // Variable-width binning.
  else {
    nBins = binning.size() - 1;
    edges = new float[nBins + 1];
    for (size_t i = 0; i <= nBins; i++) edges[i] = binning[i];
  }

}



// This is an unorthodox method of getting parameters, but cleaner in my mind
// It looks for parameter 'target' in the pset, and assumes that all entries
// have the same class (T), filling the values into a std::map.
template <class T>
void 
HLTMuonMatchAndPlot::fillMapFromPSet(map<string, T> & m, 
                                     const ParameterSet& pset, string target) 
{

  // Get the ParameterSet with name 'target' from 'pset'
  ParameterSet targetPset;
  if (pset.existsAs<ParameterSet>(target, true))        // target is tracked
    targetPset = pset.getParameterSet(target);
  else if (pset.existsAs<ParameterSet>(target, false))  // target is untracked
    targetPset = pset.getUntrackedParameterSet(target);
  
  // Get the parameter names from targetPset
  vector<string> names = targetPset.getParameterNames();
  vector<string>::const_iterator iter;
  
  for (iter = names.begin(); iter != names.end(); ++iter) {
    if (targetPset.existsAs<T>(* iter, true))           // target is tracked
      m[* iter] = targetPset.getParameter<T>(* iter);
    else if (targetPset.existsAs<T>(* iter, false))     // target is untracked
      m[* iter] = targetPset.getUntrackedParameter<T>(* iter);
  }

}



// A generic method to find the best deltaR matches from 2 collections.
template <class T1, class T2> 
vector<size_t> 
HLTMuonMatchAndPlot::matchByDeltaR(const vector<T1> & collection1, 
                                   const vector<T2> & collection2,
                                   const double maxDeltaR) 
{

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



MuonCollection
HLTMuonMatchAndPlot::selectedMuons(const MuonCollection & allMuons, 
                                   const BeamSpot & beamSpot,
                                   bool hasRecoCuts,
                                   const StringCutObjectSelector<reco::Muon> &selector,
                                   double d0Cut, double z0Cut)
{

  // If there is no selector (recoCuts does not exists), return an empty collection. 
  if (!hasRecoCuts)
    return MuonCollection();

  MuonCollection reducedMuons(allMuons);
  MuonCollection::iterator iter = reducedMuons.begin();
  while (iter != reducedMuons.end()) {
    const Track * track = 0;
    if (iter->isTrackerMuon()) track = & * iter->innerTrack();
    else if (iter->isStandAloneMuon()) track = & * iter->outerTrack();
    if (track && selector(* iter) &&
        fabs(track->dxy(beamSpot.position())) < d0Cut &&
        fabs(track->dz(beamSpot.position())) < z0Cut)
      ++iter;
    else reducedMuons.erase(iter);
  }

  return reducedMuons;

}



TriggerObjectCollection
HLTMuonMatchAndPlot::selectedTriggerObjects(
  const TriggerObjectCollection & triggerObjects,
  const TriggerEvent & triggerSummary,
  const ParameterSet & pset) 
{

  // If pset is empty, return an empty collection.
  if (!pset.exists("hltCuts"))
    return TriggerObjectCollection();

  StringCutObjectSelector<TriggerObject> selector
    (pset.getUntrackedParameter<string>("hltCuts"));

  InputTag filterTag(moduleLabels_[moduleLabels_.size() - 1], "", 
                     hltProcessName_);
  size_t filterIndex = triggerSummary.filterIndex(filterTag);

  TriggerObjectCollection selectedObjects;

  if (filterIndex < triggerSummary.sizeFilters()) {
    const Keys &keys = triggerSummary.filterKeys(filterIndex);
    for (size_t j = 0; j < keys.size(); j++ ){
      TriggerObject foundObject = triggerObjects[keys[j]];
      if (selector(foundObject))
        selectedObjects.push_back(foundObject);
    }
  }

  return selectedObjects;

}



void HLTMuonMatchAndPlot::book1D(string name, string binningType, string title)
{

  /* Properly delete the array of floats that has been allocated on
   * the heap by fillEdges.  Avoid multiple copies and internal ROOT
   * clones by simply creating the histograms directly in the DQMStore
   * using the appropriate book1D method to handle the variable bins
   * case. */ 

  size_t nBins; 
  float * edges = 0; 
  fillEdges(nBins, edges, binParams_[binningType]);

  hists_[name] = dbe_->book1D(name, title, nBins, edges);
  if (hists_[name])
    if (hists_[name]->getTH1F()->GetSumw2N())
      hists_[name]->getTH1F()->Sumw2();

  if (edges)
    delete [] edges;

}



void
HLTMuonMatchAndPlot::book2D(string name, string binningTypeX, 
                            string binningTypeY, string title) 
{
  
  /* Properly delete the arrays of floats that have been allocated on
   * the heap by fillEdges.  Avoid multiple copies and internal ROOT
   * clones by simply creating the histograms directly in the DQMStore
   * using the appropriate book2D method to handle the variable bins
   * case. */ 

  size_t  nBinsX;
  float * edgesX = 0;
  fillEdges(nBinsX, edgesX, binParams_[binningTypeX]);

  size_t  nBinsY;
  float * edgesY = 0;
  fillEdges(nBinsY, edgesY, binParams_[binningTypeY]);

  hists_[name] = dbe_->book2D(name.c_str(), title.c_str(),
			      nBinsX, edgesX, nBinsY, edgesY);
  if (hists_[name])
    if (hists_[name]->getTH2F()->GetSumw2N())
      hists_[name]->getTH2F()->Sumw2();

  if (edgesX)
    delete [] edgesX;
  if (edgesY)
    delete [] edgesY;

}


