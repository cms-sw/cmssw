// -*- C++ -*-
//
// Package:    Egamma
// Class:      HLTVars
// 
/**\class HLTVars HLTVars.cc HLTriggerOffline/Egamma/src/HLTVars.cc

 Description: Produces structs containing filter variables for HLT validation.

 Implementation:
     Should be used in combination with root script in test folder to get validation plots.
*/
//
// Original Author:  Joshua Berger
//         Created:  Wed Aug 22 20:56:48 CEST 2007
// $Id: HLTVars.cc,v 1.1 2007/09/14 19:05:49 jberger Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

#include "HLTriggerOffline/Egamma/interface/HLTVars.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidateFwd.h"

#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include "L1TriggerConfig/L1Geometry/interface/L1CaloGeometry.h"
#include "L1TriggerConfig/L1Geometry/interface/L1CaloGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/HLTReco/interface/ModuleTiming.h"
//
// class decleration
//

class HLTVars : public edm::EDProducer {
   public:
      explicit HLTVars(const edm::ParameterSet&);
      ~HLTVars();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      bool L1Match(reco::RecoEcalCandidate recoecalcand, edm::Handle<l1extra::L1EmParticleCollection> emColl, edm::ESHandle<L1CaloGeometry> l1CaloGeom);
      float HcalIsol(reco::RecoEcalCandidateRef recr, edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap);
      int ElectronPixelMatch(reco::RecoEcalCandidateRef recr, edm::Handle<reco::SeedSuperClusterAssociationCollection> barrelMap, edm::Handle<reco::SeedSuperClusterAssociationCollection> endcapMap);
      float ElectronEoverp(reco::ElectronRef eleref);
      float ElectronTrackIsol(reco::ElectronRef eleref, edm::Handle<reco::ElectronIsolationMap> depMap);
      float EcalIsol(reco::RecoEcalCandidateRef recr, edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap);
      float PhotonTrackIsol(reco::RecoEcalCandidateRef phr, edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap);
      CaloVars MCMatch(reco::SuperClusterRef scref, CaloVarsCollection mcParts);      
      // ----------member data ---------------------------
      /* For L1 Matching */
      double region_eta_size_;
      double region_eta_size_ecap_;
      double region_phi_size_;
      double barrel_end_;
      double endcap_end_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
#define PI 3.141592654
#define TWOPI 6.283185308

//
// constructors and destructor
//
HLTVars::HLTVars(const edm::ParameterSet& iConfig)
{
   //register your products
   /* 
   Store all isolation variables possible before tracking, then if tracking 
   information is available, produce another data type containing the tracking 
   variables E/p and Itrack 
   */
   produces<ElecHLTCutVarsPreTrackCollection>("SingleElecsPT");
   produces<ElecHLTCutVarsPreTrackCollection>("RelaxedSingleElecsPT");
   produces<ElecHLTCutVarsPreTrackCollection>("DoubleElecsPT");
   produces<ElecHLTCutVarsPreTrackCollection>("RelaxedDoubleElecsPT");
   produces<ElecHLTCutVarsCollection>("SingleElecs");
   produces<ElecHLTCutVarsCollection>("RelaxedSingleElecs");
   produces<ElecHLTCutVarsCollection>("DoubleElecs");
   produces<ElecHLTCutVarsCollection>("RelaxedDoubleElecs");
   produces<PhotHLTCutVarsCollection>("SinglePhots");
   produces<PhotHLTCutVarsCollection>("RelaxedSinglePhots");
   produces<PhotHLTCutVarsCollection>("DoublePhots");
   produces<PhotHLTCutVarsCollection>("RelaxedDoublePhots");
   produces<CaloVarsCollection>("mcSingleElecs");
   produces<CaloVarsCollection>("mcDoubleElecs");
   produces<CaloVarsCollection>("mcSinglePhots");
   produces<CaloVarsCollection>("mcDoublePhots");
   produces<HLTTiming>("IsoTiming");
   produces<HLTTiming>("NonIsoTiming");

   //now do what ever other initialization is needed
   //L1 Match region size
   region_eta_size_      = iConfig.getParameter<double> ("region_eta_size");
   region_eta_size_ecap_ = iConfig.getParameter<double> ("region_eta_size_ecap");
   region_phi_size_      = iConfig.getParameter<double> ("region_phi_size");
   barrel_end_           = iConfig.getParameter<double> ("barrel_end");
   endcap_end_           = iConfig.getParameter<double> ("endcap_end");
}


HLTVars::~HLTVars()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
/* These functions should calculate the variables we cut on EXACTLY. */
bool
HLTVars::L1Match(reco::RecoEcalCandidate recoecalcand, edm::Handle<l1extra::L1EmParticleCollection> emColl, edm::ESHandle<L1CaloGeometry> l1CaloGeom) 
{
  bool matched = false; 

  if(fabs(recoecalcand.eta()) < endcap_end_){
    //SC should be inside the ECAL fiducial volume
    for( l1extra::L1EmParticleCollection::const_iterator emItr = emColl->begin(); emItr != emColl->end() ;++emItr ){
      //ORCA matching method
      double etaBinLow  = 0.;
      double etaBinHigh = 0.;	
      if(fabs(recoecalcand.eta()) < barrel_end_){
        etaBinLow = emItr->eta() - region_eta_size_/2.;
        etaBinHigh = etaBinLow + region_eta_size_;
      }
      else{
	etaBinLow = emItr->eta() - region_eta_size_ecap_/2.;
	etaBinHigh = etaBinLow + region_eta_size_ecap_;
      }

      float deltaphi=fabs(recoecalcand.phi() - emItr->phi());
      if(deltaphi>TWOPI) deltaphi-=TWOPI;
      if(deltaphi>TWOPI/2.) deltaphi=TWOPI-deltaphi;

      if(recoecalcand.eta() < etaBinHigh && recoecalcand.eta() > etaBinLow &&
        deltaphi < region_phi_size_/2. )  {
        matched = true;
      }
    }
  }  
  return matched;
}

float
HLTVars::HcalIsol(reco::RecoEcalCandidateRef recr, edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap) 
{
  reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( recr );  
  float vali = mapi->val;

  return vali;
}

int
HLTVars::ElectronPixelMatch(reco::RecoEcalCandidateRef recr, edm::Handle<reco::SeedSuperClusterAssociationCollection> barrelMap, edm::Handle<reco::SeedSuperClusterAssociationCollection> endcapMap) 
{
  int nmatch = 0;
  reco::SuperClusterRef recr2 = recr->superCluster();

  for(reco::SeedSuperClusterAssociationCollection::const_iterator itb = barrelMap->begin(); itb != barrelMap->end(); itb++){
      
    edm::Ref<reco::SuperClusterCollection> theClusBarrel = itb->val;
      
    if(&(*recr2) ==  &(*theClusBarrel)) {
      nmatch++;
    }
  }

  for(reco::SeedSuperClusterAssociationCollection::const_iterator ite = endcapMap->begin(); ite != endcapMap->end(); ite++){
      
    edm::Ref<reco::SuperClusterCollection> theClusEndcap = ite->val;
      
    if(&(*recr2) ==  &(*theClusEndcap)) {
      nmatch++;
    }
  }
  return nmatch;
}

float
HLTVars::ElectronEoverp(reco::ElectronRef eleref)
{
  float elecEoverp = 0;
  const math::XYZVector trackMom =  eleref->track()->momentum();
  if( trackMom.R() != 0) elecEoverp = eleref->superCluster()->energy()/ trackMom.R();

  return elecEoverp;
}

float
HLTVars::ElectronTrackIsol(reco::ElectronRef eleref, edm::Handle<reco::ElectronIsolationMap> depMap)
{
  reco::ElectronIsolationMap::const_iterator mapi = (*depMap).find( eleref );
  float vali = mapi->val;

  return vali;
}

float 
HLTVars::EcalIsol(reco::RecoEcalCandidateRef recr, edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap)
{
  reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( recr );
  float vali = mapi->val;

  return vali;
}

float
HLTVars::PhotonTrackIsol(reco::RecoEcalCandidateRef phr, edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap)
{
  reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( phr );
  float vali = mapi->val;

  return vali;
}

CaloVars
HLTVars::MCMatch(reco::SuperClusterRef scref, CaloVarsCollection mcParts)
{
  struct CaloVars result = {0., 0., 0.};
  double minDeltaR = 0.1;
  double recoEta = scref->eta();
  double recoPhi = scref->phi();
  if (recoPhi < 0) recoPhi += TWOPI;
  for(CaloVarsCollection::const_iterator mcPart = mcParts.begin(); mcPart != mcParts.end(); ++mcPart ) {
    double mcEta = mcPart->eta;
    double mcPhi = mcPart->phi;
    if (mcPhi < 0) mcPhi += TWOPI;
    double deltaR = sqrt((mcEta - recoEta)*(mcEta - recoEta) + (mcPhi - recoPhi)*(mcPhi - recoPhi));
    if (deltaR > TWOPI) deltaR -= TWOPI;
    if (deltaR > PI) deltaR = TWOPI - deltaR;
    if (deltaR < minDeltaR) {
      minDeltaR = deltaR;
      result = *mcPart;
    }
  }

  return result;
}

// ------------ method called to produce the data  ------------
void
HLTVars::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   /* Get the L1 trigger information to see if event passes appropriate 
      filters */
   bool L1IsoSingleEG = false;
   bool L1NonIsoSingleEG = false;
   bool L1IsoDoubleEG = false;
   bool L1NonIsoDoubleEG = false;
   Handle<l1extra::L1ParticleMapCollection> L1EPM;
   try{iEvent.getByLabel("l1extraParticleMap", L1EPM);} catch(...){};
   if (L1EPM.isValid()) {
     const unsigned int nTypes = l1extra::L1ParticleMap::kNumOfL1TriggerTypes;
     for (unsigned int i = 0; i < nTypes; i++) {
       l1extra::L1ParticleMap::L1TriggerType type = static_cast<l1extra::L1ParticleMap::L1TriggerType>(i);  
       if((*L1EPM)[i].triggerDecision()) {
         if (l1extra::L1ParticleMap::triggerName(type) == "L1_SingleIsoEG12") L1IsoSingleEG = true;
         if (l1extra::L1ParticleMap::triggerName(type) == "L1_SingleEG15") L1NonIsoSingleEG = true;
         if (l1extra::L1ParticleMap::triggerName(type) == "L1_DoubleIsoEG8") L1IsoDoubleEG = true;
         if (l1extra::L1ParticleMap::triggerName(type) == "L1_DoubleEG10") L1NonIsoDoubleEG = true;
       }
     }
   }

   /* Get the particles in the event 
    * L1 trigger has two e/gamma streams: one for isolated particles which looks at nearest neighbor towers
    * to the central tower and one for non-isolated particles which did not pass the nearest neighbor cuts */
   Handle<RecoEcalCandidateCollection> l1IsoRecoEcalCands;
   Handle<RecoEcalCandidateCollection> l1NonIsoRecoEcalCands;
   Handle<l1extra::L1EmParticleCollection> l1IsoEmParts;
   Handle<l1extra::L1EmParticleCollection> l1NonIsoEmParts;

   /* Get additional information for HLT */
   ESHandle<L1CaloGeometry> l1CaloGeom;
   Handle<RecoEcalCandidateIsolationMap> l1IsoElecIHcalMap;
   Handle<RecoEcalCandidateIsolationMap> l1NonIsoElecIHcalMap;
   Handle<RecoEcalCandidateIsolationMap> l1IsoPhotIHcalMap;
   Handle<RecoEcalCandidateIsolationMap> l1NonIsoPhotIHcalMap;
   Handle<SeedSuperClusterAssociationCollection> l1IsoPixMatchBarrelMap;
   Handle<SeedSuperClusterAssociationCollection> l1IsoPixMatchEndcapMap;
   Handle<SeedSuperClusterAssociationCollection> l1NonIsoPixMatchBarrelMap;
   Handle<SeedSuperClusterAssociationCollection> l1NonIsoPixMatchEndcapMap;
   Handle<ElectronCollection> l1IsoElecs;
   Handle<ElectronCollection> l1NonIsoElecs;
   Handle<ElectronIsolationMap> l1IsoElecItrackMap;
   Handle<ElectronIsolationMap> l1NonIsoElecItrackMap;
   Handle<RecoEcalCandidateIsolationMap> l1IsoIEcalMap;
   Handle<RecoEcalCandidateIsolationMap> l1NonIsoIEcalMap;
   Handle<RecoEcalCandidateIsolationMap> l1IsoPhotItrackMap;
   Handle<RecoEcalCandidateIsolationMap> l1NonIsoPhotItrackMap;
   Handle<EventTime> hltTimes;

   try{iEvent.getByLabel("l1IsoRecoEcalCandidate", l1IsoRecoEcalCands);} catch(...){};
   try{iEvent.getByLabel("l1NonIsoRecoEcalCandidate", l1NonIsoRecoEcalCands);} catch(...){};
   try{iEvent.getByLabel("l1extraParticles", "Isolated", l1IsoEmParts);} catch(...){};
   try{iEvent.getByLabel("l1extraParticles", "NonIsolated", l1NonIsoEmParts);} catch(...){};
   try{iSetup.get<L1CaloGeometryRecord>().get(l1CaloGeom);} catch(...){};
   try{iEvent.getByLabel("l1IsolatedElectronHcalIsol", l1IsoElecIHcalMap);} catch(...){};
   try{iEvent.getByLabel("l1NonIsolatedElectronHcalIsol", l1NonIsoElecIHcalMap);} catch(...){};
   try{iEvent.getByLabel("l1IsolatedPhotonHcalIsol", l1IsoPhotIHcalMap);} catch(...){};
   try{iEvent.getByLabel("l1NonIsolatedPhotonHcalIsol", l1NonIsoPhotIHcalMap);} catch(...){};
   try{iEvent.getByLabel("l1IsoElectronPixelSeeds", "correctedHybridSuperClustersL1Isolated", l1IsoPixMatchBarrelMap);} catch(...){};
   try{iEvent.getByLabel("l1IsoElectronPixelSeeds", "correctedEndcapSuperClustersWithPreshowerL1Isolated", l1IsoPixMatchEndcapMap);} catch(...){};
   try{iEvent.getByLabel("l1NonIsoElectronPixelSeeds", "correctedHybridSuperClustersL1NonIsolated", l1NonIsoPixMatchBarrelMap);} catch(...){};
   try{iEvent.getByLabel("l1NonIsoElectronPixelSeeds", "correctedEndcapSuperClustersWithPreshowerL1NonIsolated", l1NonIsoPixMatchEndcapMap);} catch(...){};
   try{iEvent.getByLabel("pixelMatchElectronsL1IsoForHLT", l1IsoElecs);} catch(...){};
   try{iEvent.getByLabel("pixelMatchElectronsL1NonIsoForHLT", l1NonIsoElecs);} catch(...){};
   try{iEvent.getByLabel("l1IsoElectronTrackIsol", l1IsoElecItrackMap);} catch(...){};
   try{iEvent.getByLabel("l1NonIsoElectronTrackIsol", l1NonIsoElecItrackMap);} catch(...){};
   try{iEvent.getByLabel("l1IsolatedPhotonEcalIsol", l1IsoIEcalMap);} catch(...){};
   try{iEvent.getByLabel("l1NonIsolatedPhotonEcalIsol", l1NonIsoIEcalMap);} catch(...){};
   try{iEvent.getByLabel("l1IsoPhotonTrackIsol", l1IsoPhotItrackMap);} catch(...){};
   try{iEvent.getByLabel("l1NonIsoPhotonTrackIsol", l1NonIsoPhotItrackMap);} catch(...){};
   try{iEvent.getByLabel("myTimer", hltTimes);} catch(...){};

   /* Check all the labels that need to be checked */
   bool doL1Iso = l1IsoEmParts.isValid();
   bool doL1MatchIso = l1IsoRecoEcalCands.isValid();
   bool doElecIHcalIso = l1IsoElecIHcalMap.isValid();
   bool doPhotIHcalIso = l1IsoPhotIHcalMap.isValid();
   bool doPixMatchIso = l1IsoPixMatchBarrelMap.isValid() && l1IsoPixMatchEndcapMap.isValid();
   bool doEoverpIso = l1IsoElecs.isValid();
   bool doElecItrackIso = l1IsoElecItrackMap.isValid();
   bool doIEcalIso = l1IsoIEcalMap.isValid();
   bool doPhotItrackIso = l1IsoPhotItrackMap.isValid();
   bool doL1NonIso = l1NonIsoEmParts.isValid();
   bool doL1MatchNonIso = l1NonIsoRecoEcalCands.isValid();
   bool doElecIHcalNonIso = l1NonIsoElecIHcalMap.isValid();
   bool doPhotIHcalNonIso = l1NonIsoPhotIHcalMap.isValid();
   bool doPixMatchNonIso = l1NonIsoPixMatchBarrelMap.isValid() && l1NonIsoPixMatchEndcapMap.isValid();
   bool doEoverpNonIso = l1NonIsoElecs.isValid();
   bool doElecItrackNonIso = l1NonIsoElecItrackMap.isValid();
   bool doIEcalNonIso = l1NonIsoIEcalMap.isValid();
   bool doPhotItrackNonIso = l1NonIsoPhotItrackMap.isValid();

   Handle<GenParticleCandidateCollection> mcParts;
   try{iEvent.getByLabel("mcParticlesForHLT", mcParts);} catch(...){};
   
   /* Output variables */
   std::auto_ptr<ElecHLTCutVarsPreTrackCollection> SingleElecsPT(new ElecHLTCutVarsPreTrackCollection);
   std::auto_ptr<ElecHLTCutVarsPreTrackCollection> RelaxedSingleElecsPT(new ElecHLTCutVarsPreTrackCollection);
   std::auto_ptr<ElecHLTCutVarsPreTrackCollection> DoubleElecsPT(new ElecHLTCutVarsPreTrackCollection);
   std::auto_ptr<ElecHLTCutVarsPreTrackCollection> RelaxedDoubleElecsPT(new ElecHLTCutVarsPreTrackCollection);
   std::auto_ptr<ElecHLTCutVarsCollection> SingleElecs(new ElecHLTCutVarsCollection);
   std::auto_ptr<ElecHLTCutVarsCollection> RelaxedSingleElecs(new ElecHLTCutVarsCollection);
   std::auto_ptr<ElecHLTCutVarsCollection> DoubleElecs(new ElecHLTCutVarsCollection);
   std::auto_ptr<ElecHLTCutVarsCollection> RelaxedDoubleElecs(new ElecHLTCutVarsCollection);
   std::auto_ptr<PhotHLTCutVarsCollection> SinglePhots(new PhotHLTCutVarsCollection);
   std::auto_ptr<PhotHLTCutVarsCollection> RelaxedSinglePhots(new PhotHLTCutVarsCollection);
   std::auto_ptr<PhotHLTCutVarsCollection> DoublePhots(new PhotHLTCutVarsCollection);
   std::auto_ptr<PhotHLTCutVarsCollection> RelaxedDoublePhots(new PhotHLTCutVarsCollection);
   std::auto_ptr<CaloVarsCollection> mcSingleElecs(new CaloVarsCollection);
   std::auto_ptr<CaloVarsCollection> mcDoubleElecs(new CaloVarsCollection);
   std::auto_ptr<CaloVarsCollection> mcSinglePhots(new CaloVarsCollection);
   std::auto_ptr<CaloVarsCollection> mcDoublePhots(new CaloVarsCollection);
   std::auto_ptr<HLTTiming> IsoTiming(new HLTTiming);
   std::auto_ptr<HLTTiming> NonIsoTiming(new HLTTiming);

   /* Get MC particles */
   int nMCElecs = 0;
   int nMCPhots = 0;
   int nMCParts = 0;
   if (mcParts.isValid()) {
     for( GenParticleCandidateCollection::const_iterator mcpart = mcParts->begin(); mcpart != mcParts->end(); ++ mcpart ) {
       if (abs(mcpart->pdgId()) == 11 && mcpart->status() == 3) {
	 double mcEta = mcpart->eta();
	 double mcPhi = mcpart->phi();
	 if (mcPhi < 0) mcPhi += TWOPI;
	 double mcEt = mcpart->et();
	 struct CaloVars oneCutVars = { mcEt, mcEta, mcPhi };
	 mcSingleElecs->push_back(oneCutVars);
	 mcDoubleElecs->push_back(oneCutVars);
         if (fabs(mcpart->eta()) < 2.5 && (fabs(mcpart->eta()) < 1.4442 || fabs(mcpart->eta()) > 1.566)) {
           nMCElecs++; // Tells us how many MC electrons pass eta cut in order to check that filter
         }
       }
       if (abs(mcpart->pdgId()) == 22 && mcpart->status() == 3) {
         double mcEta = mcpart->eta();
         double mcPhi = mcpart->phi();
         if (mcPhi < 0) mcPhi += TWOPI;
         double mcEt = mcpart->et();
         struct CaloVars oneCutVars = { mcEt, mcEta, mcPhi };
         mcSinglePhots->push_back(oneCutVars);
         mcDoublePhots->push_back(oneCutVars);
         if (fabs(mcpart->eta()) < 2.5 && (fabs(mcpart->eta()) < 1.4442 || fabs(mcpart->eta()) > 1.566)) {
           nMCPhots++;
         }
       }
       if (mcpart->status() == 3) {
	 if (fabs(mcpart->eta()) < 2.5 && (fabs(mcpart->eta()) < 1.4442 || fabs(mcpart->eta()) > 1.566)) {
	   nMCParts++;
	 }
       }
     }
   }

   double l1MatchIsoTime = 0.;
   double EtIsoTime = 0.;
   double ElecIHcalIsoTime = 0.;
   double pixMatchIsoTime = 0.;
   double EoverpIsoTime = 0.;
   double ElecItrackIsoTime = 0.;
   double l1MatchNonIsoTime = 0.;
   double EtNonIsoTime = 0.;
   double ElecIHcalNonIsoTime = 0.;
   double pixMatchNonIsoTime = 0.;
   double EoverpNonIsoTime = 0.;
   double ElecItrackNonIsoTime = 0.;
   double IEcalIsoTime = 0.;
   double PhotIHcalIsoTime = 0.;
   double PhotItrackIsoTime = 0.;
   double IEcalNonIsoTime = 0.;
   double PhotIHcalNonIsoTime = 0.;
   double PhotItrackNonIsoTime = 0.;

   for (unsigned int i = 0; i < hltTimes->size(); i++) {
     if (hltTimes->name(i) == "ecalPreshowerDigis" || 
         hltTimes->name(i) == "ecalRegionalEgammaFEDs" || 
         hltTimes->name(i) == "ecalRegionalEgammaDigis" || 
         hltTimes->name(i) == "ecalRegionalEgammaWeightUncalibRecHit" || 
         hltTimes->name(i) == "ecalRegionalEgammaRecHit" ||
         hltTimes->name(i) == "ecalPreshowerRecHit" ||
         hltTimes->name(i) == "hltIslandBasicClustersEndcapL1Isolated" ||
         hltTimes->name(i) == "hltIslandBasicClustersBarrelL1Isolated" ||
         hltTimes->name(i) == "hltHybridSuperClustersL1Isolated" ||
         hltTimes->name(i) == "hltIslandSuperClustersL1Isolated" ||
	 hltTimes->name(i) == "correctedIslandEndcapSuperClustersL1Isolated" ||
	 hltTimes->name(i) == "correctedIslandBarrelSuperClustersL1Isolated" ||
	 hltTimes->name(i) == "correctedHybridSuperClustersL1Isolated" ||
	 hltTimes->name(i) == "correctedEndcapSuperClustersWithPreshowerL1Isolated" ||
	 hltTimes->name(i) == "l1IsoRecoEcalCandidate") 
     {
       l1MatchIsoTime += hltTimes->time(i);
       EtIsoTime += hltTimes->time(i);
       ElecIHcalIsoTime += hltTimes->time(i);
       pixMatchIsoTime += hltTimes->time(i);
       EoverpIsoTime += hltTimes->time(i);
       ElecItrackIsoTime += hltTimes->time(i);
       l1MatchNonIsoTime += hltTimes->time(i);
       EtNonIsoTime += hltTimes->time(i);
       ElecIHcalNonIsoTime += hltTimes->time(i);
       pixMatchNonIsoTime += hltTimes->time(i);
       EoverpNonIsoTime += hltTimes->time(i);
       ElecItrackNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "hltIslandBasicClustersEndcapL1NonIsolated" ||
         hltTimes->name(i) == "hltIslandBasicClustersBarrelL1NonIsolated" ||
         hltTimes->name(i) == "hltHybridSuperClustersL1NonIsolated" ||
         hltTimes->name(i) == "hltIslandSuperClustersL1NonIsolated" ||
	 hltTimes->name(i) == "correctedIslandEndcapSuperClustersL1NonIsolated" ||
	 hltTimes->name(i) == "correctedIslandBarrelSuperClustersL1NonIsolated" ||
	 hltTimes->name(i) == "correctedHybridSuperClustersL1NonIsolated" ||
	 hltTimes->name(i) == "correctedEndcapSuperClustersWithPreshowerL1NonIsolated" ||
	 hltTimes->name(i) == "l1NonIsoRecoEcalCandidate") 
     {
       l1MatchNonIsoTime += hltTimes->time(i);
       EtNonIsoTime += hltTimes->time(i);
       ElecIHcalNonIsoTime += hltTimes->time(i);
       pixMatchNonIsoTime += hltTimes->time(i);
       EoverpNonIsoTime += hltTimes->time(i);
       ElecItrackNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "hcalZeroSuppressedDigis" ||
         hltTimes->name(i) == "hbhereco" ||
         hltTimes->name(i) == "hfreco" ||
         hltTimes->name(i) == "l1IsolatedElectronHcalIsol")
     {
       ElecIHcalIsoTime += hltTimes->time(i);
       ElecIHcalNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "l1NonIsolatedElectronHcalIsol")
     {
       ElecIHcalNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "siPixelDigis" ||
         hltTimes->name(i) == "siPixelClusters" ||
         hltTimes->name(i) == "siPixelRecHits" ||
         hltTimes->name(i) == "l1IsoElectronPixelSeeds")
     {
       pixMatchIsoTime += hltTimes->time(i);
       EoverpIsoTime += hltTimes->time(i);
       ElecItrackIsoTime += hltTimes->time(i);
       pixMatchNonIsoTime += hltTimes->time(i);
       EoverpNonIsoTime += hltTimes->time(i);
       ElecItrackNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "l1NonIsoElectronPixelSeeds")
     {
       pixMatchNonIsoTime += hltTimes->time(i);
       EoverpNonIsoTime += hltTimes->time(i);
       ElecItrackNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "SiStripRawToClustersFacility" ||
	 hltTimes->name(i) == "siStripClusters" ||
	 hltTimes->name(i) == "ckfL1IsoTrackCandidatesBarrel" ||
         hltTimes->name(i) == "ckfL1IsoTrackCandidatesEndcap" ||
         hltTimes->name(i) == "ctfL1IsoWithMaterialTracksBarrel" ||
         hltTimes->name(i) == "ctfL1IsoWithMaterialTracksEndcap" ||
         hltTimes->name(i) == "pixelMatchElectronsL1IsoForHLT")
     {
       EoverpIsoTime += hltTimes->time(i);
       ElecItrackIsoTime += hltTimes->time(i);
       EoverpNonIsoTime += hltTimes->time(i);
       ElecItrackNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "ckfL1NonIsoTrackCandidatesBarrel" ||
         hltTimes->name(i) == "ckfL1NonIsoTrackCandidatesEndcap" ||
         hltTimes->name(i) == "ctfL1NonIsoWithMaterialTracksBarrel" ||
         hltTimes->name(i) == "ctfL1NonIsoWithMaterialTracksEndcap" ||
         hltTimes->name(i) == "pixelMatchElectronsL1NonIsoForHLT")
     {
       EoverpNonIsoTime += hltTimes->time(i);
       ElecItrackNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "l1IsoElectronsRegionalPixelSeedGenerator" ||
         hltTimes->name(i) == "l1IsoElectronsRegionalCkfTrackCandidates" ||
         hltTimes->name(i) == "l1IsoElectronsRegionalCTFFinalFitWithMaterial" ||
         hltTimes->name(i) == "l1IsoElectronTrackIsol")
     {
       ElecItrackIsoTime += hltTimes->time(i);
       ElecItrackNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "l1NonIsoElectronsRegionalPixelSeedGenerator" ||
         hltTimes->name(i) == "l1NonIsoElectronsRegionalCkfTrackCandidates" ||
         hltTimes->name(i) == "l1NonIsoElectronsRegionalCTFFinalFitWithMaterial" ||
         hltTimes->name(i) == "l1NonIsoElectronTrackIsol")
     {
       ElecItrackNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "l1IsolatedPhotonEcalIsol")
     {
       IEcalIsoTime += hltTimes->time(i);
       IEcalNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "l1NonIsolatedPhotonEcalIsol")
     {
       IEcalNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "hcalZeroSuppressedDigis" ||
         hltTimes->name(i) == "hbhereco" ||
         hltTimes->name(i) == "hfreco" ||
         hltTimes->name(i) == "l1IsolatedPhotonHcalIsol")
     {
       PhotIHcalIsoTime += hltTimes->time(i);
       PhotIHcalNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "l1NonIsolatedPhotonHcalIsol")
     {
       PhotIHcalNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "siPixelDigis" ||
         hltTimes->name(i) == "siPixelClusters" ||
         hltTimes->name(i) == "siPixelRecHits" ||
         hltTimes->name(i) == "SiStripRawToClustersFacility" ||
         hltTimes->name(i) == "siStripClusters" ||
         hltTimes->name(i) == "l1IsoEgammaRegionalPixelSeedGenerator" ||
         hltTimes->name(i) == "l1IsoEgammaRegionalCkfTrackCandidates" ||
         hltTimes->name(i) == "l1IsoEgammaRegionalCTFFinalFitWithMaterial" ||
	 hltTimes->name(i) == "l1IsoPhotonTrackIsol")
     {
       PhotItrackIsoTime += hltTimes->time(i);
       PhotItrackNonIsoTime += hltTimes->time(i);
     }
     if (hltTimes->name(i) == "l1NonIsoEgammaRegionalPixelSeedGenerator" ||
         hltTimes->name(i) == "l1NonIsoEgammaRegionalCkfTrackCandidates" ||
         hltTimes->name(i) == "l1NonIsoEgammaRegionalCTFFinalFitWithMaterial" ||
	 hltTimes->name(i) == "l1NonIsoPhotonTrackIsol")
     {
       PhotItrackNonIsoTime += hltTimes->time(i);
     }
   }
   struct HLTTiming IsoTimingTemp = {l1MatchIsoTime, EtIsoTime, ElecIHcalIsoTime, pixMatchIsoTime, EoverpIsoTime, ElecItrackIsoTime, IEcalIsoTime, PhotIHcalIsoTime, PhotItrackIsoTime};
   *IsoTiming = IsoTimingTemp;
struct HLTTiming NonIsoTimingTemp = {l1MatchNonIsoTime, EtNonIsoTime, ElecIHcalNonIsoTime, pixMatchNonIsoTime, EoverpNonIsoTime, ElecItrackNonIsoTime, IEcalNonIsoTime, PhotIHcalNonIsoTime, PhotItrackNonIsoTime};
 *NonIsoTiming = NonIsoTimingTemp;

   /* Maybe turn this code into a function later?  Actually calculate and store isolation variables for Iso and NonIso
      The code is identical, so it might be good to consolidate it into a function soon */
   if (doL1MatchIso) {  
     for (RecoEcalCandidateCollection::const_iterator recoecalcand = l1IsoRecoEcalCands->begin(); recoecalcand != l1IsoRecoEcalCands->end(); ++recoecalcand) {
       RecoEcalCandidateRef recr(RecoEcalCandidateRef(l1IsoRecoEcalCands, recoecalcand - l1IsoRecoEcalCands->begin()));
       SuperClusterRef recr2 = recr->superCluster();
       bool l1Match = false;
       if (doL1MatchIso) l1Match = L1Match(*recoecalcand, l1IsoEmParts, l1CaloGeom);
       double Et = recoecalcand->et();
       float ElecIHcal = 99999.;
       if (doElecIHcalIso) ElecIHcal = HcalIsol(recr, l1IsoElecIHcalMap);
       int pixMatch = 0;
       if (doPixMatchIso) pixMatch = ElectronPixelMatch(recr, l1IsoPixMatchBarrelMap, l1IsoPixMatchEndcapMap);
       float IEcal = 99999.;
       if (doIEcalIso) IEcal = EcalIsol(recr, l1IsoIEcalMap);
       float PhotIHcal = 99999.;
       if (doPhotIHcalIso) PhotIHcal = HcalIsol(recr, l1IsoPhotIHcalMap);
       float PhotItrack = 99999.;
       if (doPhotItrackIso) PhotItrack = PhotonTrackIsol(recr, l1IsoPhotItrackMap);
       float Eoverp = 99999.;
       float ElecItrack = 99999.;
       double eta = recoecalcand->eta();
       double phi = recoecalcand->phi();
       CaloVars matchElec = MCMatch(recr2, *mcSingleElecs);
       CaloVars matchPhot = MCMatch(recr2, *mcSinglePhots);
       if (phi < 0) phi += TWOPI;
       struct ElecHLTCutVarsPreTrack elecStructPT = {l1Match, Et, ElecIHcal, pixMatch, eta, phi, matchElec.Et, matchElec.eta, matchElec.phi}; 
       struct PhotHLTCutVars photStruct = {l1Match, Et, IEcal, PhotIHcal, PhotItrack, eta, phi, matchPhot.Et, matchPhot.eta, matchPhot.phi};
       SingleElecsPT->push_back(elecStructPT);
       RelaxedSingleElecsPT->push_back(elecStructPT);
       DoubleElecsPT->push_back(elecStructPT);
       RelaxedDoubleElecsPT->push_back(elecStructPT);
       SinglePhots->push_back(photStruct);
       RelaxedSinglePhots->push_back(photStruct);
       DoublePhots->push_back(photStruct);
       RelaxedDoublePhots->push_back(photStruct);
       if(doEoverpIso) { 
         for (ElectronCollection::const_iterator eleccand = l1IsoElecs->begin(); eleccand != l1IsoElecs->end(); ++eleccand) {
           ElectronRef electronref(ElectronRef(l1IsoElecs, eleccand - l1IsoElecs->begin()));
           const SuperClusterRef theClus = electronref->superCluster();
           if (&(*recr2) == &(*theClus)) {
             Eoverp = ElectronEoverp(electronref);
             if (doElecItrackIso) ElecItrack = ElectronTrackIsol(electronref, l1IsoElecItrackMap);
	     eta = electronref->eta();
 	     phi = electronref->phi();
             if (phi < 0) phi += TWOPI;
             struct ElecHLTCutVars elecStruct = {l1Match, Et, ElecIHcal, pixMatch, Eoverp, ElecItrack, eta, phi, matchElec.Et, matchElec.eta, matchElec.phi};
             SingleElecs->push_back(elecStruct);
             RelaxedSingleElecs->push_back(elecStruct);
             DoubleElecs->push_back(elecStruct);
             RelaxedDoubleElecs->push_back(elecStruct);
	     Eoverp = 99999.;
	     ElecItrack = 99999.;
	     eta = recoecalcand->eta();
	     phi = recoecalcand->phi();
	   }
	 }
       }
     } 
   }

   if(doL1MatchNonIso) {
     for (RecoEcalCandidateCollection::const_iterator recoecalcand = l1NonIsoRecoEcalCands->begin(); recoecalcand != l1NonIsoRecoEcalCands->end(); ++recoecalcand) {
       RecoEcalCandidateRef recr(RecoEcalCandidateRef(l1NonIsoRecoEcalCands, recoecalcand - l1NonIsoRecoEcalCands->begin()));
       SuperClusterRef recr2 = recr->superCluster();
       bool l1Match = false;
       if (doL1MatchNonIso) l1Match = L1Match(*recoecalcand, l1NonIsoEmParts, l1CaloGeom);
       double Et = recoecalcand->et();
       float ElecIHcal = 99999.;
       if (doElecIHcalNonIso) ElecIHcal = HcalIsol(recr, l1NonIsoElecIHcalMap);
       int pixMatch = 0;
       if (doPixMatchNonIso) pixMatch = ElectronPixelMatch(recr, l1NonIsoPixMatchBarrelMap, l1NonIsoPixMatchEndcapMap);
       float IEcal = 99999.;
       if (doIEcalNonIso) IEcal = EcalIsol(recr, l1NonIsoIEcalMap);
       float PhotIHcal = 99999.;
       if (doPhotIHcalNonIso) PhotIHcal = HcalIsol(recr, l1NonIsoPhotIHcalMap);
       float PhotItrack = 99999.;
       if (doPhotItrackNonIso) PhotItrack = PhotonTrackIsol(recr, l1NonIsoPhotItrackMap);
       float Eoverp = 99999.;
       float ElecItrack = 99999.;
       double eta = recoecalcand->eta();
       double phi = recoecalcand->phi(); 
       CaloVars matchElec = MCMatch(recr2, *mcSingleElecs);
       CaloVars matchPhot = MCMatch(recr2, *mcSinglePhots);
       if (phi < 0) phi += TWOPI;
       struct ElecHLTCutVarsPreTrack elecStructPT = {l1Match, Et, ElecIHcal, pixMatch, eta, phi, matchElec.Et, matchElec.eta, matchElec.phi};
       struct PhotHLTCutVars photStruct = {l1Match, Et, IEcal, PhotIHcal, PhotItrack, eta, phi, matchPhot.Et, matchPhot.eta, matchPhot.phi};
       RelaxedSingleElecsPT->push_back(elecStructPT);
       RelaxedDoubleElecsPT->push_back(elecStructPT);
       RelaxedSinglePhots->push_back(photStruct);
       RelaxedDoublePhots->push_back(photStruct);
       if(doEoverpNonIso) {
         for (ElectronCollection::const_iterator eleccand = l1NonIsoElecs->begin(); eleccand != l1NonIsoElecs->end(); ++eleccand) {
           ElectronRef electronref(ElectronRef(l1NonIsoElecs, eleccand - l1NonIsoElecs->begin()));
           const reco::SuperClusterRef theClus = electronref->superCluster();
           if (&(*recr2) == &(*theClus)) {
             Eoverp = ElectronEoverp(electronref);
             if (doElecItrackNonIso) ElecItrack = ElectronTrackIsol(electronref, l1NonIsoElecItrackMap);
	     eta = electronref->eta();
	     phi = electronref->phi();
             if (phi < 0) phi += TWOPI;
             struct ElecHLTCutVars elecStruct = {l1Match, Et, ElecIHcal, pixMatch, Eoverp, ElecItrack, eta, phi, matchElec.Et, matchElec.eta, matchElec.phi};
             RelaxedSingleElecs->push_back(elecStruct);
             RelaxedDoubleElecs->push_back(elecStruct);
	     Eoverp = 99999.;
	     ElecItrack = 99999.;
	     eta = recoecalcand->eta();
	     phi = recoecalcand->phi();
           }
	 }
       }
     }
   }

   if (nMCElecs >= 1 && L1IsoSingleEG) iEvent.put(SingleElecsPT, "SingleElecsPT"); 
   if (nMCElecs >= 1 && L1NonIsoSingleEG) iEvent.put(RelaxedSingleElecsPT, "RelaxedSingleElecsPT"); 
   if (nMCElecs >= 2 && L1IsoDoubleEG) iEvent.put(DoubleElecsPT, "DoubleElecsPT"); 
   if (nMCElecs >= 2 && L1NonIsoDoubleEG) iEvent.put(RelaxedDoubleElecsPT, "RelaxedDoubleElecsPT"); 

   if (nMCElecs >= 1 && L1IsoSingleEG) iEvent.put(SingleElecs, "SingleElecs"); 
   if (nMCElecs >= 1 && L1NonIsoSingleEG) iEvent.put(RelaxedSingleElecs, "RelaxedSingleElecs"); 
   if (nMCElecs >= 2 && L1IsoDoubleEG) iEvent.put(DoubleElecs, "DoubleElecs"); 
   if (nMCElecs >= 2 && L1NonIsoDoubleEG) iEvent.put(RelaxedDoubleElecs, "RelaxedDoubleElecs"); 

   if (nMCParts >= 1 && L1IsoSingleEG) iEvent.put(SinglePhots, "SinglePhots"); 
   if (nMCParts >= 1 && L1NonIsoSingleEG) iEvent.put(RelaxedSinglePhots, "RelaxedSinglePhots"); 
   if (nMCParts >= 2 && L1IsoDoubleEG) iEvent.put(DoublePhots, "DoublePhots"); 
   if (nMCParts >= 2 && L1NonIsoDoubleEG) iEvent.put(RelaxedDoublePhots, "RelaxedDoublePhots"); 

   if (nMCElecs >= 1) iEvent.put(mcSingleElecs, "mcSingleElecs");
   if (nMCElecs >= 2) iEvent.put(mcDoubleElecs, "mcDoubleElecs");
   if (nMCPhots >= 1) iEvent.put(mcSinglePhots, "mcSinglePhots");
   if (nMCPhots >= 2) iEvent.put(mcDoublePhots, "mcDoublePhots");

   iEvent.put(IsoTiming, "IsoTiming");
   iEvent.put(NonIsoTiming, "NonIsoTiming");
}

// ------------ method called once each job just before starting event loop  ------------
void 
HLTVars::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTVars::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTVars);
