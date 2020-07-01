/** \class HLTMuonTurnOnAnalyzer
 *  Get L1/HLT turn on curves
 *
 *  \author J. Alcaraz
 */

#include "HLTrigger/Muon/test/HLTMuonTurnOnAnalyzer.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TFile.h"
#include "TH1F.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;
using namespace l1extra;

/// Constructor
HLTMuonTurnOnAnalyzer::HLTMuonTurnOnAnalyzer(const ParameterSet& pset) {
  theGenLabel = pset.getUntrackedParameter<InputTag>("GenLabel");
  useMuonFromGenerator = pset.getUntrackedParameter<bool>("UseMuonFromGenerator");
  theL1CollectionLabel = pset.getUntrackedParameter<InputTag>("L1CollectionLabel");
  theHLTCollectionLabels = pset.getUntrackedParameter<std::vector<InputTag> >("HLTCollectionLabels");
  theGenToken = consumes<edm::HepMCProduct>(theGenLabel);
  theL1CollectionToken = consumes<trigger::TriggerFilterObjectWithRefs>(theL1CollectionLabel);
  for (auto& theHLTCollectionLabel : theHLTCollectionLabels) {
    theHLTCollectionTokens.push_back(consumes<trigger::TriggerFilterObjectWithRefs>(theHLTCollectionLabel));
  }
  theReferenceThreshold = pset.getUntrackedParameter<double>("ReferenceThreshold");

  thePtMin = pset.getUntrackedParameter<double>("PtMin");
  thePtMax = pset.getUntrackedParameter<double>("PtMax");
  theNbins = pset.getUntrackedParameter<unsigned int>("Nbins");

  theRootFileName = pset.getUntrackedParameter<string>("RootFileName");

  theNumberOfEvents = 0.;
}

/// Destructor
HLTMuonTurnOnAnalyzer::~HLTMuonTurnOnAnalyzer() = default;

void HLTMuonTurnOnAnalyzer::beginJob() {
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  char chname[256];
  char chtitle[256];
  snprintf(chname, 255, "eff_%s", theL1CollectionLabel.encode().c_str());
  snprintf(chtitle, 255, "Efficiency (%%) vs Generated Pt (GeV), label=%s", theL1CollectionLabel.encode().c_str());
  hL1eff = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  hL1nor = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);

  for (auto& theHLTCollectionLabel : theHLTCollectionLabels) {
    snprintf(chname, 255, "eff_%s", theHLTCollectionLabel.encode().c_str());
    snprintf(chtitle, 255, "Efficiency (%%) vs Generated Pt (GeV), label=%s", theHLTCollectionLabel.encode().c_str());
    hHLTeff.push_back(new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax));
    hHLTnor.push_back(new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax));
  }
}

void HLTMuonTurnOnAnalyzer::endJob() {
  LogInfo("HLTMuonTurnOnAnalyzer") << " (Weighted) number of analyzed events= " << theNumberOfEvents;
  theFile->cd();

  if (theNumberOfEvents == 0) {
    LogInfo("HLTMuonTurnOnAnalyzer") << " No histograms will be written because number of events=0!!!";
    theFile->Close();
    return;
  }

  // L1 operations
  hL1eff->Divide(hL1nor);
  hL1eff->Scale(100.);

  // HLT operations
  for (unsigned int i = 0; i < theHLTCollectionLabels.size(); i++) {
    hHLTeff[i]->Divide(hHLTnor[i]);
    hHLTeff[i]->Scale(100.);
  }

  // Write the histos to file
  hL1eff->Write();
  for (unsigned int i = 0; i < theHLTCollectionLabels.size(); i++) {
    hHLTeff[i]->Write();
  }

  theFile->Close();
}

void HLTMuonTurnOnAnalyzer::analyze(const Event& event, const EventSetup& eventSetup) {
  theFile->cd();

  // Get the HepMC product
  double this_event_weight = 1.;
  Handle<HepMCProduct> genProduct;
  event.getByToken(theGenToken, genProduct);

  const HepMC::GenEvent* evt = genProduct->GetEvent();
  HepMC::WeightContainer weights = evt->weights();
  if (!weights.empty())
    this_event_weight = weights[0];
  theNumberOfEvents += this_event_weight;

  // Get the L1 collection
  Handle<TriggerFilterObjectWithRefs> l1cands;
  event.getByToken(theL1CollectionToken, l1cands);

  // Get the HLT collections
  std::vector<Handle<TriggerFilterObjectWithRefs> > hltcands(theHLTCollectionLabels.size());

  unsigned int modules_in_this_event = 0;
  for (unsigned int i = 0; i < theHLTCollectionLabels.size(); i++) {
    event.getByToken(theHLTCollectionTokens[i], hltcands[i]);
    if (hltcands[i].failedToGet())
      break;
    modules_in_this_event++;
  }

  // Get the muon with maximum pt at generator level or reconstruction, depending on the choice
  bool refmuon_found = false;
  double ptuse = -1;

  if (useMuonFromGenerator) {
    HepMC::GenEvent::particle_const_iterator part;
    for (part = evt->particles_begin(); part != evt->particles_end(); ++part) {
      int id1 = (*part)->pdg_id();
      if (id1 != 13 && id1 != -13)
        continue;
      float pt1 = (*part)->momentum().perp();
      if (pt1 > ptuse) {
        refmuon_found = true;
        ptuse = pt1;
      }
    }
  } else {
    unsigned int i = modules_in_this_event - 1;
    vector<RecoChargedCandidateRef> vref;
    hltcands[i]->getObjects(TriggerMuon, vref);
    for (auto& k : vref) {
      RecoChargedCandidateRef candref = RecoChargedCandidateRef(k);
      TrackRef tk = candref->get<TrackRef>();
      double pt = tk->pt();
      if (pt > ptuse) {
        refmuon_found = true;
        ptuse = pt;
      }
    }
  }

  if (!refmuon_found) {
    LogInfo("HLTMuonTurnOnAnalyzer") << " NO reference muon found!!!";
    LogInfo("HLTMuonTurnOnAnalyzer") << " Skipping event";
    return;
  }

  // Fix L1 thresholds to obtain the efficiecy plot
  unsigned int nL1FoundRef = 0;
  double epsilon = 0.001;
  vector<L1MuonParticleRef> l1mu;
  l1cands->getObjects(TriggerL1Mu, l1mu);
  for (auto& k : l1mu) {
    L1MuonParticleRef candref = L1MuonParticleRef(k);
    // L1 PTs are "quantized" due to LUTs.
    // Their meaning: true_pt > ptLUT more than 90% pof the times
    double ptLUT = candref->pt();
    // Add "epsilon" to avoid rounding errors when ptLUT==L1Threshold
    if (ptLUT + epsilon > theReferenceThreshold)
      nL1FoundRef++;
  }
  hL1nor->Fill(ptuse, this_event_weight);
  if (nL1FoundRef > 0)
    hL1eff->Fill(ptuse, this_event_weight);

  // HLT filling
  unsigned int last_module = modules_in_this_event - 1;
  if ((!useMuonFromGenerator) && last_module > 0)
    last_module--;
  for (unsigned int i = 0; i <= last_module; i++) {
    double ptcut = theReferenceThreshold;
    unsigned nFound = 0;
    vector<RecoChargedCandidateRef> vref;
    hltcands[i]->getObjects(TriggerMuon, vref);
    for (auto& k : vref) {
      RecoChargedCandidateRef candref = RecoChargedCandidateRef(k);
      TrackRef tk = candref->get<TrackRef>();
      double pt = tk->pt();
      if (pt > ptcut)
        nFound++;
    }
    hHLTnor[i]->Fill(ptuse, this_event_weight);
    if (nFound > 0)
      hHLTeff[i]->Fill(ptuse, this_event_weight);
  }
}

DEFINE_FWK_MODULE(HLTMuonTurnOnAnalyzer);
