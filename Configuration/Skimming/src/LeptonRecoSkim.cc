
// -*- C++ -*-
//
// Package:    LeptonRecoSkim
// Class:      LeptonRecoSkim
//
/**\class LeptonRecoSkim LeptonRecoSkim.cc Configuration/Skimming/src/LeptonRecoSkim.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Massimiliano Chiorboli,40 4-A01,+41227671535,
//         Created:  Wed Mar 31 21:49:08 CEST 2010
// $Id: LeptonRecoSkim.cc,v 1.1 2010/11/05 18:37:51 torimoto Exp $
//
//

#include "Configuration/Skimming/interface/LeptonRecoSkim.h"

using namespace edm;
using namespace reco;
using namespace std;

//
// constructors and destructor
//
LeptonRecoSkim::LeptonRecoSkim(const edm::ParameterSet& iConfig)
    : m_CaloGeoToken(esConsumes()),
      m_CaloTopoToken(esConsumes()),
      hltLabel(iConfig.getParameter<edm::InputTag>("HltLabel")),
      filterName(iConfig.getParameter<std::string>("@module_label")),
      gsfElectronCollectionToken_(consumes(iConfig.getParameter<edm::InputTag>("electronCollection"))),
      pfCandidateCollectionToken_(consumes(iConfig.getParameter<edm::InputTag>("pfElectronCollection"))),
      muonCollectionToken_(consumes(iConfig.getParameter<edm::InputTag>("muonCollection"))),
      caloJetCollectionToken_(consumes(iConfig.getParameter<edm::InputTag>("caloJetCollection"))),
      pfJetCollectionToken_(consumes(iConfig.getParameter<edm::InputTag>("PFJetCollection"))),
      ebRecHitCollectionToken_(consumes(iConfig.getParameter<edm::InputTag>("ecalBarrelRecHitsCollection"))),
      eeRecHitCollectionToken_(consumes(iConfig.getParameter<edm::InputTag>("ecalEndcapRecHitsCollection"))),
      useElectronSelection(iConfig.getParameter<bool>("UseElectronSelection")),
      usePfElectronSelection(iConfig.getParameter<bool>("UsePfElectronSelection")),
      useMuonSelection(iConfig.getParameter<bool>("UseMuonSelection")),
      useHtSelection(iConfig.getParameter<bool>("UseHtSelection")),
      usePFHtSelection(iConfig.getParameter<bool>("UsePFHtSelection")),
      ptElecMin(iConfig.getParameter<double>("electronPtMin")),
      ptPfElecMin(iConfig.getParameter<double>("pfElectronPtMin")),
      nSelectedElectrons(iConfig.getParameter<int>("electronN")),
      nSelectedPfElectrons(iConfig.getParameter<int>("pfElectronN")),
      ptGlobalMuonMin(iConfig.getParameter<double>("globalMuonPtMin")),
      ptTrackerMuonMin(iConfig.getParameter<double>("trackerMuonPtMin")),
      nSelectedMuons(iConfig.getParameter<int>("muonN")),
      htMin(iConfig.getParameter<double>("HtMin")),
      pfHtMin(iConfig.getParameter<double>("PFHtMin")),
      htJetThreshold(iConfig.getParameter<double>("HtJetThreshold")),
      pfHtJetThreshold(iConfig.getParameter<double>("PFHtJetThreshold")) {
  NeventsTotal = 0;
  NeventsFiltered = 0;
  NHltMu9 = 0;
  NHltDiMu3 = 0;

  NtotalElectrons = 0;
  NmvaElectrons = 0;
}

LeptonRecoSkim::~LeptonRecoSkim() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool LeptonRecoSkim::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool accept = false;

  NeventsTotal++;

  ElectronCutPassed = false;
  PfElectronCutPassed = false;
  MuonCutPassed = false;
  HtCutPassed = false;
  PFHtCutPassed = false;

  //  edm::Handle<TriggerResults> trhv;
  //  iEvent.getByLabel(hltLabel,trhv);

  //  const edm::TriggerNames& triggerNames_ = iEvent.triggerNames(*trhv);

  //  if(trhv->at(triggerNames_.triggerIndex("HLT_Mu9")).accept()) NHltMu9++;
  //  if(trhv->at(triggerNames_.triggerIndex("HLT_DoubleMu3")).accept()) NHltDiMu3++;

  this->handleObjects(iEvent, iSetup);

  if (useElectronSelection) {
    int nElecPassingCut = 0;
    for (unsigned int i = 0; i < theElectronCollection->size(); i++) {
      GsfElectron electron = (*theElectronCollection)[i];
      //      if (electron.ecalDrivenSeed()) {
      float elpt = electron.pt();
      //        if (electron.sigmaIetaIeta() < 0.002) continue;
      if (elpt > ptElecMin) {
        NtotalElectrons++;
        nElecPassingCut++;
        if (electron.mva_e_pi() > -0.1)
          NmvaElectrons++;
      }
      LogDebug("LeptonRecoSkim") << "elpt = " << elpt << endl;
      // } // closes if (electron.ecalDrivenSeed()) {
    }
    if (nElecPassingCut >= nSelectedElectrons)
      ElectronCutPassed = true;
  } else
    ElectronCutPassed = true;

  if (usePfElectronSelection) {
    int nPfElecPassingCut = 0;
    for (unsigned int i = 0; i < thePfCandidateCollection->size(); i++) {
      const reco::PFCandidate& thePfCandidate = (*thePfCandidateCollection)[i];
      if (thePfCandidate.particleId() != reco::PFCandidate::e)
        continue;
      if (thePfCandidate.gsfTrackRef().isNull())
        continue;
      float pfelpt = thePfCandidate.pt();
      //        if (electron.sigmaIetaIeta() < 0.002) continue;
      if (pfelpt > ptPfElecMin)
        nPfElecPassingCut++;
      LogDebug("LeptonRecoSkim") << "pfelpt = " << pfelpt << endl;
    }
    if (nPfElecPassingCut >= nSelectedPfElectrons)
      PfElectronCutPassed = true;
  } else
    PfElectronCutPassed = true;

  if (useMuonSelection) {
    int nMuonPassingCut = 0;
    for (unsigned int i = 0; i < theMuonCollection->size(); i++) {
      Muon muon = (*theMuonCollection)[i];
      if (!(muon.isGlobalMuon() || muon.isTrackerMuon()))
        continue;
      const TrackRef siTrack = muon.innerTrack();
      const TrackRef globalTrack = muon.globalTrack();
      float muonpt;
      float ptMuonMin;
      //      if (siTrack.isNonnull() && globalTrack.isNonnull()) {
      if (muon.isGlobalMuon() && muon.isTrackerMuon()) {
        muonpt = max(siTrack->pt(), globalTrack->pt());
        ptMuonMin = ptGlobalMuonMin;
      } else if (muon.isGlobalMuon() && !(muon.isTrackerMuon())) {
        muonpt = globalTrack->pt();
        ptMuonMin = ptGlobalMuonMin;
      } else if (muon.isTrackerMuon() && !(muon.isGlobalMuon())) {
        muonpt = siTrack->pt();
        ptMuonMin = ptTrackerMuonMin;
      } else {
        muonpt = 0;  // if we get here this is a STA only muon
        ptMuonMin = 999999.;
      }
      if (muonpt > ptMuonMin)
        nMuonPassingCut++;
      LogDebug("RecoSelectorCuts") << "muonpt = " << muonpt << endl;
    }
    if (nMuonPassingCut >= nSelectedMuons)
      MuonCutPassed = true;
  } else
    MuonCutPassed = true;

  if (useHtSelection) {
    double Ht = 0;
    for (unsigned int i = 0; i < theCaloJetCollection->size(); i++) {
      //      if((*theCaloJetCollection)[i].eta()<2.6 && (*theCaloJetCollection)[i].emEnergyFraction() <= 0.01) continue;
      if ((*theCaloJetCollection)[i].pt() > htJetThreshold)
        Ht += (*theCaloJetCollection)[i].pt();
    }
    if (Ht > htMin)
      HtCutPassed = true;
  } else
    HtCutPassed = true;

  if (usePFHtSelection) {
    double PFHt = 0;
    for (unsigned int i = 0; i < thePFJetCollection->size(); i++) {
      if ((*thePFJetCollection)[i].pt() > pfHtJetThreshold)
        PFHt += (*thePFJetCollection)[i].pt();
    }
    if (PFHt > pfHtMin)
      PFHtCutPassed = true;
  } else
    PFHtCutPassed = true;

  if (PfElectronCutPassed && ElectronCutPassed && MuonCutPassed && HtCutPassed && PFHtCutPassed)
    accept = true;

  if (accept)
    NeventsFiltered++;

  firstEvent = false;
  return accept;
}

// ------------ method called once each job just before starting event loop  ------------
void LeptonRecoSkim::beginJob() { firstEvent = true; }

// ------------ method called once each job just after ending the event loop  ------------
void LeptonRecoSkim::endJob() {
  edm::LogPrint("LeptonRecoSkim") << "Filter Name            = " << filterName << endl;
  edm::LogPrint("LeptonRecoSkim") << "Total number of events = " << NeventsTotal << endl;
  edm::LogPrint("LeptonRecoSkim") << "Total HLT_Mu9          = " << NHltMu9 << endl;
  edm::LogPrint("LeptonRecoSkim") << "Total HLT_DoubleMu3    = " << NHltDiMu3 << endl;
  edm::LogPrint("LeptonRecoSkim") << "Filtered events        = " << NeventsFiltered << endl;
  edm::LogPrint("LeptonRecoSkim") << "Filter Efficiency      = " << (float)NeventsFiltered / (float)NeventsTotal
                                  << endl;
  edm::LogPrint("LeptonRecoSkim") << endl;
  edm::LogPrint("LeptonRecoSkim") << "N total electrons      = " << NtotalElectrons << endl;
  edm::LogPrint("LeptonRecoSkim") << "N mva>-0.1 electrons   = " << NmvaElectrons << endl;
  edm::LogPrint("LeptonRecoSkim") << endl;
  edm::LogPrint("LeptonRecoSkim") << endl;
}

void LeptonRecoSkim::handleObjects(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Get the electrons
  theElectronCollection = &iEvent.get(gsfElectronCollectionToken_);

  //Get the pf electrons
  thePfCandidateCollection = &iEvent.get(pfCandidateCollectionToken_);

  //Get the Muons
  theMuonCollection = &iEvent.get(muonCollectionToken_);

  //Get the CaloJets
  theCaloJetCollection = &iEvent.get(caloJetCollectionToken_);

  //Get the PfJets
  thePFJetCollection = &iEvent.get(pfJetCollectionToken_);

  // Get the ECAL rechhits to clean the spikes
  // Get EB RecHits
  theEcalBarrelCollection = &iEvent.get(ebRecHitCollectionToken_);
  // Get EE RecHits
  theEcalEndcapCollection = &iEvent.get(eeRecHitCollectionToken_);

  // Get topology for spike cleaning
  theCaloGeometry = &iSetup.getData(m_CaloGeoToken);
  theCaloTopology = &iSetup.getData(m_CaloTopoToken);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LeptonRecoSkim);
