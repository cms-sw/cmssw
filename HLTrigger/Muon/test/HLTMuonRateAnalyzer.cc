/** \class HLTMuonRateAnalyzer
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author J. Alcaraz
 */

#include "HLTrigger/Muon/test/HLTMuonRateAnalyzer.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
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
HLTMuonRateAnalyzer::HLTMuonRateAnalyzer(const ParameterSet& pset) {
  theGenLabel = pset.getUntrackedParameter<InputTag>("GenLabel");
  theL1CollectionLabel = pset.getUntrackedParameter<InputTag>("L1CollectionLabel");
  theHLTCollectionLabels = pset.getUntrackedParameter<std::vector<InputTag> >("HLTCollectionLabels");
  theGenToken = consumes<edm::HepMCProduct>(theGenLabel);
  theL1CollectionToken = consumes<trigger::TriggerFilterObjectWithRefs>(theL1CollectionLabel);
  for (auto& theHLTCollectionLabel : theHLTCollectionLabels) {
    theHLTCollectionTokens.push_back(consumes<trigger::TriggerFilterObjectWithRefs>(theHLTCollectionLabel));
  }
  theL1ReferenceThreshold = pset.getUntrackedParameter<double>("L1ReferenceThreshold");
  theNSigmas = pset.getUntrackedParameter<std::vector<double> >("NSigmas90");

  theNumberOfObjects = pset.getUntrackedParameter<unsigned int>("NumberOfObjects");
  theCrossSection = pset.getUntrackedParameter<double>("CrossSection");
  // Convert it already into /nb/s)
  theLuminosity = pset.getUntrackedParameter<double>("Luminosity") * 1.e-33;

  thePtMin = pset.getUntrackedParameter<double>("PtMin");
  thePtMax = pset.getUntrackedParameter<double>("PtMax");
  theNbins = pset.getUntrackedParameter<unsigned int>("Nbins");

  theRootFileName = pset.getUntrackedParameter<string>("RootFileName");

  theNumberOfEvents = 0.;
}

/// Destructor
HLTMuonRateAnalyzer::~HLTMuonRateAnalyzer() = default;

void HLTMuonRateAnalyzer::beginJob() {
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  char chname[256];
  char chtitle[256];
  snprintf(chname, 255, "eff_%s", theL1CollectionLabel.encode().c_str());
  snprintf(chtitle, 255, "Efficiency (%%) vs L1 Pt threshold (GeV), label=%s", theL1CollectionLabel.encode().c_str());
  hL1eff = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  snprintf(chname, 255, "rate_%s", theL1CollectionLabel.encode().c_str());
  snprintf(chtitle,
           255,
           "Rate (Hz) vs L1 Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})",
           theL1CollectionLabel.encode().c_str(),
           theLuminosity * 1.e33);
  hL1rate = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);

  for (auto& theHLTCollectionLabel : theHLTCollectionLabels) {
    snprintf(chname, 255, "eff_%s", theHLTCollectionLabel.encode().c_str());
    snprintf(
        chtitle, 255, "Efficiency (%%) vs HLT Pt threshold (GeV), label=%s", theHLTCollectionLabel.encode().c_str());
    hHLTeff.push_back(new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax));
    snprintf(chname, 255, "rate_%s", theHLTCollectionLabel.encode().c_str());
    snprintf(chtitle,
             255,
             "Rate (Hz) vs HLT Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})",
             theHLTCollectionLabel.encode().c_str(),
             theLuminosity * 1.e33);
    hHLTrate.push_back(new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax));
  }
}

void HLTMuonRateAnalyzer::endJob() {
  LogInfo("HLTMuonRateAnalyzer") << " (Weighted) number of analyzed events= " << theNumberOfEvents;
  theFile->cd();

  if (theNumberOfEvents == 0) {
    LogInfo("HLTMuonRateAnalyzer") << " No histograms will be written because number of events=0!!!";
    theFile->Close();
    return;
  }

  // L1 operations
  for (unsigned int k = 0; k <= theNbins + 1; k++) {
    double this_eff = hL1eff->GetBinContent(k) / theNumberOfEvents;
    // Hope that this will be essentially OK for weighted samples
    // It should be strictly OK in a binomial scheme when weights = 1
    double this_eff_error = hL1eff->GetBinError(k) / theNumberOfEvents * sqrt(1 - this_eff);
    hL1eff->SetBinContent(k, 100 * this_eff);
    hL1eff->SetBinError(k, 100 * this_eff_error);
    double this_rate = theLuminosity * theCrossSection * this_eff;
    double this_rate_error = theLuminosity * theCrossSection * this_eff_error;
    hL1rate->SetBinContent(k, this_rate);
    hL1rate->SetBinError(k, this_rate_error);
  }

  // HLT operations
  for (unsigned int i = 0; i < theHLTCollectionLabels.size(); i++) {
    for (unsigned int k = 0; k <= theNbins + 1; k++) {
      // Hope that this will be essentially OK for weighted samples
      // It should be strictly OK in a binomial scheme when weights = 1
      double this_eff = hHLTeff[i]->GetBinContent(k) / theNumberOfEvents;
      double this_eff_error = hHLTeff[i]->GetBinError(k) / theNumberOfEvents;
      hHLTeff[i]->SetBinContent(k, 100 * this_eff);
      hHLTeff[i]->SetBinError(k, 100 * this_eff_error);
      double this_rate = theLuminosity * theCrossSection * this_eff;
      double this_rate_error = theLuminosity * theCrossSection * this_eff_error;
      hHLTrate[i]->SetBinContent(k, this_rate);
      hHLTrate[i]->SetBinError(k, this_rate_error);
    }
  }

  // Write the histos to file
  hL1eff->Write();
  hL1rate->Write();
  for (unsigned int i = 0; i < theHLTCollectionLabels.size(); i++) {
    hHLTeff[i]->Write();
    hHLTrate[i]->Write();
  }

  theFile->Close();
}

void HLTMuonRateAnalyzer::analyze(const Event& event, const EventSetup& eventSetup) {
  theFile->cd();

  // Get the HepMC product
  double this_event_weight = 1.;
  try {
    Handle<HepMCProduct> genProduct;
    event.getByToken(theGenToken, genProduct);
    const HepMC::GenEvent* evt = genProduct->GetEvent();
    HepMC::WeightContainer weights = evt->weights();
    if (!weights.empty())
      this_event_weight = weights[0];
  } catch (...) {
    LogInfo("HLTMuonRateAnalyzer") << " NO HepMCProduct found!!!!!!!!!!!!!!!";
    LogInfo("HLTMuonRateAnalyzer") << " SETTING EVENT WEIGHT TO 1";
  }
  theNumberOfEvents += this_event_weight;

  // Get the L1 collection
  Handle<TriggerFilterObjectWithRefs> l1cands;
  event.getByToken(theL1CollectionToken, l1cands);
  if (l1cands.failedToGet())
    return;

  // Get the HLT collections
  std::vector<Handle<TriggerFilterObjectWithRefs> > hltcands(theHLTCollectionLabels.size());
  unsigned int modules_in_this_event = 0;
  for (unsigned int i = 0; i < theHLTCollectionLabels.size(); i++) {
    event.getByToken(theHLTCollectionTokens[i], hltcands[i]);
    if (hltcands[i].failedToGet())
      break;
    modules_in_this_event++;
  }

  // Fix L1 thresholds to obtain HLT plots
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
    if (ptLUT + epsilon > theL1ReferenceThreshold)
      nL1FoundRef++;
  }

  for (unsigned int j = 0; j < theNbins; j++) {
    double ptcut = thePtMin + j * (thePtMax - thePtMin) / theNbins;

    // L1 filling
    unsigned int nFound = 0;
    for (auto& k : l1mu) {
      L1MuonParticleRef candref = L1MuonParticleRef(k);
      double pt = candref->pt();
      if (pt > ptcut)
        nFound++;
    }
    if (nFound >= theNumberOfObjects)
      hL1eff->Fill(ptcut, this_event_weight);

    // Stop here if L1 reference cuts were not satisfied
    if (nL1FoundRef < theNumberOfObjects)
      continue;

    // HLT filling
    for (unsigned int i = 0; i < modules_in_this_event; i++) {
      unsigned nFound = 0;
      vector<RecoChargedCandidateRef> vref;
      hltcands[i]->getObjects(TriggerMuon, vref);
      for (auto& k : vref) {
        RecoChargedCandidateRef candref = RecoChargedCandidateRef(k);
        TrackRef tk = candref->get<TrackRef>();
        double pt = tk->pt();
        double err0 = tk->error(0);
        double abspar0 = fabs(tk->parameter(0));
        // convert to 90% efficiency threshold
        if (abspar0 > 0)
          pt += theNSigmas[i] * err0 / abspar0 * pt;
        if (pt > ptcut)
          nFound++;
      }
      if (nFound >= theNumberOfObjects) {
        hHLTeff[i]->Fill(ptcut, this_event_weight);
      } else {
        break;
      }
    }
  }
}

DEFINE_FWK_MODULE(HLTMuonRateAnalyzer);
