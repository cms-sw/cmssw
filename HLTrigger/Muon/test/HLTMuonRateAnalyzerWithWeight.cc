/** \class HLTMuonRateAnalyzerWithWeight
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author J. Alcaraz
 */

#include "HLTrigger/Muon/test/HLTMuonRateAnalyzerWithWeight.h"

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
HLTMuonRateAnalyzerWithWeight::HLTMuonRateAnalyzerWithWeight(const ParameterSet& pset) {
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

  // Convert it already into /nb/s)
  theLuminosity = pset.getUntrackedParameter<double>("Luminosity") * 1.e-33;
  theIntegratedLumi = pset.getParameter<double>("IntLumi");
  type = pset.getParameter<unsigned int>("Type");

  thePtMin = pset.getUntrackedParameter<double>("PtMin");
  thePtMax = pset.getUntrackedParameter<double>("PtMax");
  theNbins = pset.getUntrackedParameter<unsigned int>("Nbins");

  theRootFileName = pset.getUntrackedParameter<string>("RootFileName");
  theNumberOfBCEvents = 0.;
  theNumberOfLightEvents = 0.;
}

/// Destructor
HLTMuonRateAnalyzerWithWeight::~HLTMuonRateAnalyzerWithWeight() = default;

void HLTMuonRateAnalyzerWithWeight::beginJob() {
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();
  hNumEvents = new TH1F("NumEvents", "Number of Events analyzed", 2, -0.5, 1.5);

  char chname[256];
  char chtitle[256];
  snprintf(chname, 255, "Lighteff_%s", theL1CollectionLabel.encode().c_str());
  snprintf(chtitle,
           255,
           "Light Quark events Efficiency (%%) vs L1 Pt threshold (GeV), label=%s",
           theL1CollectionLabel.encode().c_str());
  hLightL1eff = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  hLightL1eff->Sumw2();
  snprintf(chname, 255, "Lightrate_%s", theL1CollectionLabel.encode().c_str());
  snprintf(chtitle,
           255,
           "Light Quark events Rate (Hz) vs L1 Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})",
           theL1CollectionLabel.encode().c_str(),
           theLuminosity * 1.e33);
  hLightL1rate = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  hLightL1rate->Sumw2();
  snprintf(chname, 255, "BCeff_%s", theL1CollectionLabel.encode().c_str());
  snprintf(chtitle,
           255,
           "BC Quark events Efficiency (%%) vs L1 Pt threshold (GeV), label=%s",
           theL1CollectionLabel.encode().c_str());
  hBCL1eff = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  hBCL1eff->Sumw2();
  snprintf(chname, 255, "BCrate_%s", theL1CollectionLabel.encode().c_str());
  snprintf(chtitle,
           255,
           "BC Quark events Rate (Hz) vs L1 Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})",
           theL1CollectionLabel.encode().c_str(),
           theLuminosity * 1.e33);
  hBCL1rate = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  hBCL1rate->Sumw2();

  for (unsigned int i = 0; i < theHLTCollectionLabels.size(); i++) {
    snprintf(chname, 255, "Lighteff_%s", theHLTCollectionLabels[i].encode().c_str());
    snprintf(chtitle,
             255,
             "Light Quark events Efficiency (%%) vs HLT Pt threshold (GeV), label=%s",
             theHLTCollectionLabels[i].encode().c_str());
    hLightHLTeff.push_back(new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax));
    hLightHLTeff[i]->Sumw2();
    snprintf(chname, 255, "Light rate_%s", theHLTCollectionLabels[i].encode().c_str());
    snprintf(chtitle,
             255,
             "Light Quark events Rate (Hz) vs HLT Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})",
             theHLTCollectionLabels[i].encode().c_str(),
             theLuminosity * 1.e33);
    hLightHLTrate.push_back(new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax));
    hLightHLTrate[i]->Sumw2();
    snprintf(chname, 255, "BCeff_%s", theHLTCollectionLabels[i].encode().c_str());
    snprintf(chtitle,
             255,
             "BC Quark events Efficiency (%%) vs HLT Pt threshold (GeV), label=%s",
             theHLTCollectionLabels[i].encode().c_str());
    hBCHLTeff.push_back(new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax));
    hBCHLTeff[i]->Sumw2();
    snprintf(chname, 255, "BC rate_%s", theHLTCollectionLabels[i].encode().c_str());
    snprintf(chtitle,
             255,
             "BC Quark events Rate (Hz) vs HLT Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})",
             theHLTCollectionLabels[i].encode().c_str(),
             theLuminosity * 1.e33);
    hBCHLTrate.push_back(new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax));
    hBCHLTrate[i]->Sumw2();
  }
}

void HLTMuonRateAnalyzerWithWeight::endJob() {
  //std::cout << "in endjob"<<endl;
  theFile->cd();
  //std::cout << "in file"<<endl;
  if (theNumberOfBCEvents == 0 && theNumberOfLightEvents == 0) {
    theFile->Close();
    return;
  }
  //std::cout << "we have events"<<endl;

  // L1 operations
  for (unsigned int k = 0; k <= theNbins + 1; k++) {
    if (theNumberOfLightEvents != 0) {
      double this_eff = hLightL1eff->GetBinContent(k) / theNumberOfLightEvents;
      double this_eff_error = hLightL1eff->GetBinError(k) / theNumberOfLightEvents * sqrt(1 - this_eff);
      hLightL1eff->SetBinContent(k, 100 * this_eff);
      hLightL1eff->SetBinError(k, 100 * this_eff_error);
      double this_rate = theLuminosity * this_eff * theNumberOfLightEvents / theIntegratedLumi;
      double this_rate_error = theLuminosity * this_eff_error * theNumberOfLightEvents / theIntegratedLumi;
      hLightL1rate->SetBinContent(k, this_rate);
      hLightL1rate->SetBinError(k, this_rate_error);
    }
    if (theNumberOfBCEvents != 0) {
      double this_eff = hBCL1eff->GetBinContent(k) / theNumberOfBCEvents;
      double this_eff_error = hBCL1eff->GetBinError(k) / theNumberOfBCEvents * sqrt(1 - this_eff);
      hBCL1eff->SetBinContent(k, 100 * this_eff);
      hBCL1eff->SetBinError(k, 100 * this_eff_error);
      double this_rate = theLuminosity * this_eff * theNumberOfBCEvents / theIntegratedLumi;
      double this_rate_error = theLuminosity * this_eff_error * theNumberOfBCEvents / theIntegratedLumi;
      hBCL1rate->SetBinContent(k, this_rate);
      hBCL1rate->SetBinError(k, this_rate_error);
    }
  }

  // HLT operations
  for (unsigned int i = 0; i < theHLTCollectionLabels.size(); i++) {
    for (unsigned int k = 0; k <= theNbins + 1; k++) {
      // Hope that this will be essentially OK for weighted samples
      // It should be strictly OK in a binomial scheme when weights = 1
      if (theNumberOfLightEvents != 0) {
        double this_eff = hLightHLTeff[i]->GetBinContent(k) / theNumberOfLightEvents;
        double this_eff_error = hLightHLTeff[i]->GetBinError(k) / theNumberOfLightEvents;
        hLightHLTeff[i]->SetBinContent(k, 100 * this_eff);
        hLightHLTeff[i]->SetBinError(k, 100 * this_eff_error);
        double this_rate = theLuminosity * this_eff * theNumberOfLightEvents / theIntegratedLumi;
        double this_rate_error = theLuminosity * this_eff_error * theNumberOfLightEvents / theIntegratedLumi;
        hLightHLTrate[i]->SetBinContent(k, this_rate);
        hLightHLTrate[i]->SetBinError(k, this_rate_error);
      }
      if (theNumberOfBCEvents != 0) {
        double this_eff = hBCHLTeff[i]->GetBinContent(k) / theNumberOfBCEvents;
        double this_eff_error = hBCHLTeff[i]->GetBinError(k) / theNumberOfBCEvents;
        hBCHLTeff[i]->SetBinContent(k, 100 * this_eff);
        hBCHLTeff[i]->SetBinError(k, 100 * this_eff_error);
        double this_rate = theLuminosity * this_eff * theNumberOfBCEvents / theIntegratedLumi;
        double this_rate_error = theLuminosity * this_eff_error * theNumberOfBCEvents / theIntegratedLumi;
        hBCHLTrate[i]->SetBinContent(k, this_rate);
        hBCHLTrate[i]->SetBinError(k, this_rate_error);
      }
    }
  }

  // Write the histos to file
  hNumEvents->Write();
  hLightL1eff->Write();
  hLightL1rate->Write();
  hBCL1eff->Write();
  hBCL1rate->Write();
  for (unsigned int i = 0; i < theHLTCollectionLabels.size(); i++) {
    hLightHLTeff[i]->Write();
    hLightHLTrate[i]->Write();
    hBCHLTeff[i]->Write();
    hBCHLTrate[i]->Write();
  }
  theFile->Close();
}

void HLTMuonRateAnalyzerWithWeight::analyze(const Event& event, const EventSetup& eventSetup) {
  theFile->cd();

  // Get the HepMC product
  double this_event_weight = 1.;
  bool bcevent = false;
  try {
    Handle<HepMCProduct> genProduct;
    event.getByToken(theGenToken, genProduct);
    const HepMC::GenEvent* evt = genProduct->GetEvent();
    HepMC::WeightContainer weights = evt->weights();
    bcevent = isbc(*evt);
    hNumEvents->Fill(1. * bcevent);
    if (!weights.empty())
      this_event_weight = weights[0];
    if (type == 3)
      this_event_weight *= parentWeight(*evt);
    LogInfo("HLTMuonRateAnalyzerWithWeight") << " This event weight is " << this_event_weight;
  } catch (...) {
    LogWarning("HLTMuonRateAnalyzerWithWeight") << " NO HepMCProduct found!!!!!!!!!!!!!!!";
    LogWarning("HLTMuonRateAnalyzerWithWeight") << " SETTING EVENT WEIGHT TO 1";
  }
  if (bcevent)
    theNumberOfBCEvents += this_event_weight;
  else
    theNumberOfLightEvents += this_event_weight;
  // Get the L1 collection
  Handle<TriggerFilterObjectWithRefs> l1cands;
  event.getByToken(theL1CollectionToken, l1cands);
  if (l1cands.failedToGet()) {
    LogInfo("HLTMuonRateAnalyzerWithWeight") << " No L1 collection";
    // Do nothing
    return;
  }

  // Get the HLT collections
  std::vector<Handle<TriggerFilterObjectWithRefs> > hltcands(theHLTCollectionLabels.size());

  unsigned int modules_in_this_event = 0;
  for (unsigned int i = 0; i < theHLTCollectionLabels.size(); i++) {
    event.getByToken(theHLTCollectionTokens[i], hltcands[i]);
    if (hltcands[i].failedToGet()) {
      LogInfo("HLTMuonRateAnalyzerWithWeight") << " No " << theHLTCollectionLabels[i];
      break;
    }
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
    if (nFound >= theNumberOfObjects) {
      if (bcevent)
        hBCL1eff->Fill(ptcut, this_event_weight);
      else
        hLightL1eff->Fill(ptcut, this_event_weight);
    }
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
        if (bcevent)
          hBCHLTeff[i]->Fill(ptcut, this_event_weight);
        else
          hLightHLTeff[i]->Fill(ptcut, this_event_weight);
      } else {
        break;
      }
    }
  }
}

bool HLTMuonRateAnalyzerWithWeight::isbc(HepMC::GenEvent const& Gevt) {
  bool mybc = false;
  int npart = 0;
  int nb = 0;
  int nc = 0;
  for (HepMC::GenEvent::particle_const_iterator particle = Gevt.particles_begin(); particle != Gevt.particles_end();
       ++particle) {
    ++npart;
    int id = abs((*particle)->pdg_id());
    //	int status=(*particle)->status();
    if (id == 5 || id == 4) {
      if (npart == 6 || npart == 7) {
        mybc = true;
        break;
      } else {
        HepMC::GenVertex* parent = (*particle)->production_vertex();
        for (auto ic = parent->particles_in_const_begin(); ic != parent->particles_in_const_end(); ic++) {
          int pid = (*ic)->pdg_id();
          if (pid == 21 && id == 5)
            nb++;
          else if (pid == 21 && id == 4)
            nc++;
        }
      }
    }
  }
  if (nb > 1 || nc > 1)
    mybc = true;
  return mybc;
}

double HLTMuonRateAnalyzerWithWeight::parentWeight(HepMC::GenEvent const& Gevt) {
  double AdditionalWeight = 1.;
  if (type != 3)
    return AdditionalWeight;
  for (HepMC::GenEvent::particle_const_iterator particle = Gevt.particles_begin(); particle != Gevt.particles_end();
       ++particle) {
    int id = abs((*particle)->pdg_id());
    double pt = (*particle)->momentum().perp();
    if (id == 13 && pt > 10) {
      HepMC::GenVertex* parent = (*particle)->production_vertex();
      for (auto ic = parent->particles_in_const_begin(); ic != parent->particles_in_const_end(); ic++) {
        int apid = abs((*ic)->pdg_id());
        LogInfo("HLTMuonRateAnalyzerWithWeight") << " Absolute parent id is " << apid;
        if (apid > 10000)
          apid = apid - (apid / 10000) * 10000;
        if (apid > 1000)
          apid /= 1000;
        if (apid > 100 && apid != 130)
          apid /= 100;
        LogInfo("HLTMuonRateAnalyzerWithWeight") << " It will be treated as " << apid;
        if (apid == 5)
          AdditionalWeight = 1. / 8.4;  //b mesons
        else if (apid == 4)
          AdditionalWeight = 1. / 6.0;  //c mesons
        else if (apid == 15)
          AdditionalWeight = 1. / 8.7;  //taus
        else if (apid == 3 || apid == 130)
          AdditionalWeight = 1. / 7.3;  //s-mesons
        else if (apid == 2)
          AdditionalWeight = 1. / 0.8;  //pions
      }
    }
  }
  return AdditionalWeight;
}

DEFINE_FWK_MODULE(HLTMuonRateAnalyzerWithWeight);
