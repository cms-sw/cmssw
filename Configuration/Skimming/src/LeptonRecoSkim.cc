
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
// $Id: LeptonRecoSkim.cc,v 1.2 2010/11/18 01:04:23 wmtan Exp $
//
//


#include "Configuration/Skimming/interface/LeptonRecoSkim.h"


using namespace edm;
using namespace reco;
using namespace std;

//
// constructors and destructor
//
LeptonRecoSkim::LeptonRecoSkim(const edm::ParameterSet& iConfig):
  hltLabel(iConfig.getParameter<edm::InputTag>("HltLabel")),
  filterName(iConfig.getParameter<std::string>("@module_label")),
  m_electronSrc(iConfig.getParameter<edm::InputTag>("electronCollection")),
  m_pfelectronSrc(iConfig.getParameter<edm::InputTag>("pfElectronCollection")),
  m_muonSrc(iConfig.getParameter<edm::InputTag>("muonCollection")),
  m_jetsSrc(iConfig.getParameter<edm::InputTag>("caloJetCollection")),
  m_pfjetsSrc(iConfig.getParameter<edm::InputTag>("PFJetCollection")),
  m_ebRecHitsSrc(iConfig.getParameter<edm::InputTag>("ecalBarrelRecHitsCollection")),
  m_eeRecHitsSrc(iConfig.getParameter<edm::InputTag>("ecalEndcapRecHitsCollection")),
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
  pfHtJetThreshold(iConfig.getParameter<double>("PFHtJetThreshold"))
{
  NeventsTotal = 0;
  NeventsFiltered = 0;
  NHltMu9 = 0;
  NHltDiMu3 = 0;

  NtotalElectrons = 0;
  NmvaElectrons = 0;
}


LeptonRecoSkim::~LeptonRecoSkim()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
LeptonRecoSkim::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


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

  this->handleObjects(iEvent,iSetup);

  if(useElectronSelection) {
    int nElecPassingCut = 0;
    for(unsigned int i=0; i<theElectronCollection->size(); i++) {
      GsfElectron electron = (*theElectronCollection)[i];
      //      if (electron.ecalDrivenSeed()) {
	float elpt = electron.pt();
	//        if (electron.sigmaIetaIeta() < 0.002) continue;
	if(elpt>ptElecMin)  {
	  NtotalElectrons++;
	  nElecPassingCut++;
	  if(electron.mva()>-0.1) NmvaElectrons++;
	}
	LogDebug("LeptonRecoSkim") << "elpt = " << elpt << endl;
	// } // closes if (electron.ecalDrivenSeed()) {
    }
    if(nElecPassingCut>=nSelectedElectrons) ElectronCutPassed = true;
  }
  else ElectronCutPassed = true;
  
   if(usePfElectronSelection) {
     int nPfElecPassingCut = 0;
     for(unsigned int i=0; i<thePfCandidateCollection->size(); i++) {
       const reco::PFCandidate& thePfCandidate = (*thePfCandidateCollection)[i];
       if(thePfCandidate.particleId()!=reco::PFCandidate::e) continue;
       if(thePfCandidate.gsfTrackRef().isNull()) continue;
       float pfelpt = thePfCandidate.pt();
       //        if (electron.sigmaIetaIeta() < 0.002) continue;
       if(pfelpt>ptPfElecMin)  nPfElecPassingCut++;
       LogDebug("LeptonRecoSkim") << "pfelpt = " << pfelpt << endl;
     }
     if(nPfElecPassingCut>=nSelectedPfElectrons) PfElectronCutPassed = true;
   }
   else PfElectronCutPassed = true;
  


  if(useMuonSelection) {
    int nMuonPassingCut = 0;
    for(unsigned int i=0; i<theMuonCollection->size(); i++) {
      Muon muon = (*theMuonCollection)[i];
      if(! (muon.isGlobalMuon() || muon.isTrackerMuon()) ) continue;
      const TrackRef siTrack     = muon.innerTrack();
      const TrackRef globalTrack = muon.globalTrack();
      float muonpt;
      float ptMuonMin;
      //      if (siTrack.isNonnull() && globalTrack.isNonnull()) {
      if (muon.isGlobalMuon() &&  muon.isTrackerMuon()) {
	muonpt = max(siTrack->pt(), globalTrack->pt());
	ptMuonMin = ptGlobalMuonMin;
      }
      else if (muon.isGlobalMuon()  &&  !(muon.isTrackerMuon())) {
	muonpt = globalTrack->pt();
	ptMuonMin = ptGlobalMuonMin;
      }
      else if (muon.isTrackerMuon()  &&  !(muon.isGlobalMuon())) {
 	muonpt = siTrack->pt();
	ptMuonMin = ptTrackerMuonMin;
      }
      else {
	muonpt = 0; // if we get here this is a STA only muon
	ptMuonMin = 999999.;
      }
      if(muonpt>ptMuonMin) nMuonPassingCut++;
      LogDebug("RecoSelectorCuts") << "muonpt = " << muonpt << endl;
    }
    if(nMuonPassingCut>=nSelectedMuons) MuonCutPassed = true;
  }
  else MuonCutPassed = true;

  if(useHtSelection) {
    double Ht = 0;
    for(unsigned int i=0; i<theCaloJetCollection->size(); i++) {
      //      if((*theCaloJetCollection)[i].eta()<2.6 && (*theCaloJetCollection)[i].emEnergyFraction() <= 0.01) continue;
      if((*theCaloJetCollection)[i].pt()>htJetThreshold) Ht += (*theCaloJetCollection)[i].pt();
    }
    if(Ht>htMin) HtCutPassed = true;
  }
  else HtCutPassed = true;

  if(usePFHtSelection) {
    double PFHt = 0;
    for(unsigned int i=0; i<thePFJetCollection->size(); i++) {
      if((*thePFJetCollection)[i].pt()>pfHtJetThreshold) PFHt += (*thePFJetCollection)[i].pt();
    }
    if(PFHt>pfHtMin) PFHtCutPassed = true;
  }
  else PFHtCutPassed = true;


  if(PfElectronCutPassed      && 
     ElectronCutPassed        && 
     MuonCutPassed            && 
     HtCutPassed              &&
     PFHtCutPassed            ) accept = true;
  
  if(accept) NeventsFiltered++;

  firstEvent = false;
  return accept;
  


}

// ------------ method called once each job just before starting event loop  ------------
void 
LeptonRecoSkim::beginJob()
{
  firstEvent = true;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LeptonRecoSkim::endJob() {
  cout << "Filter Name            = " << filterName                                       << endl;
  cout << "Total number of events = " << NeventsTotal                                     << endl;
  cout << "Total HLT_Mu9          = " << NHltMu9                                          << endl;
  cout << "Total HLT_DoubleMu3    = " << NHltDiMu3                                        << endl;
  cout << "Filtered events        = " << NeventsFiltered                                  << endl;
  cout << "Filter Efficiency      = " << (float)NeventsFiltered / (float)NeventsTotal     << endl;
  cout << endl;
  cout << "N total electrons      = " << NtotalElectrons << endl;
  cout << "N mva>-0.1 electrons   = " << NmvaElectrons << endl;
  cout << endl;
  cout << endl;
}




void LeptonRecoSkim::handleObjects(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //Get the electrons
  Handle<GsfElectronCollection> theElectronCollectionHandle; 
  iEvent.getByLabel(m_electronSrc, theElectronCollectionHandle);
  theElectronCollection = theElectronCollectionHandle.product();

  //Get the pf electrons
  Handle<reco::PFCandidateCollection> thePfCandidateHandle;
  iEvent.getByLabel(m_pfelectronSrc, thePfCandidateHandle);
  thePfCandidateCollection = thePfCandidateHandle.product();

  //Get the Muons
  Handle<MuonCollection> theMuonCollectionHandle; 
  iEvent.getByLabel(m_muonSrc, theMuonCollectionHandle);
  theMuonCollection = theMuonCollectionHandle.product();

  //Get the CaloJets
  Handle<CaloJetCollection> theCaloJetCollectionHandle;
  iEvent.getByLabel(m_jetsSrc, theCaloJetCollectionHandle);
  theCaloJetCollection = theCaloJetCollectionHandle.product();

  //Get the PfJets
  Handle<PFJetCollection> thePFJetCollectionHandle;
  iEvent.getByLabel(m_pfjetsSrc, thePFJetCollectionHandle);
  thePFJetCollection = thePFJetCollectionHandle.product();

  //Get the ECAL rechhits to clean the spikes
// Get EB RecHits
  edm::Handle<EcalRecHitCollection> ebRecHitsHandle;
  iEvent.getByLabel(m_ebRecHitsSrc, ebRecHitsHandle);
  theEcalBarrelCollection = ebRecHitsHandle.product();
// Get EE RecHits
  edm::Handle<EcalRecHitCollection> eeRecHitsHandle;
  iEvent.getByLabel(m_eeRecHitsSrc, eeRecHitsHandle);
  theEcalEndcapCollection = eeRecHitsHandle.product();

// Get topology for spike cleaning
   edm::ESHandle<CaloGeometry> geometryHandle;
   iSetup.get<CaloGeometryRecord>().get(geometryHandle);
   theCaloGeometry = geometryHandle.product();
//   theCaloBarrelSubdetTopology = new EcalBarrelTopology(geometryHandle);
//   theCaloEndcapSubdetTopology = new EcalEndcapTopology(geometryHandle);

  edm::ESHandle<CaloTopology> pTopology;
  iSetup.get<CaloTopologyRecord>().get(pTopology);
  theCaloTopology = pTopology.product();
  
  


}
    
    


//define this as a plug-in
DEFINE_FWK_MODULE(LeptonRecoSkim);
