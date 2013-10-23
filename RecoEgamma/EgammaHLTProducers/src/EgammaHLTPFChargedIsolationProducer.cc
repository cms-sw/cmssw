/** 
 *
 *  \author Matteo Sani (UCSD)
 * 
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTPFChargedIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <DataFormats/Math/interface/deltaR.h>

EgammaHLTPFChargedIsolationProducer::EgammaHLTPFChargedIsolationProducer(const edm::ParameterSet& config) {

  electronProducer_          = config.getParameter<edm::InputTag>("electronProducer");
  pfCandidateProducer_       = config.getParameter<edm::InputTag>("pfCandidatesProducer");
  recoEcalCandidateProducer_ = config.getParameter<edm::InputTag>("recoEcalCandidateProducer"); 
  beamSpotProducer_          = config.getParameter<edm::InputTag>("beamSpotProducer");

  useGsfTrack_ = config.getParameter<bool>("useGsfTrack");
  useSCRefs_ = config.getParameter<bool>("useSCRefs");
  
  drMax_ = config.getParameter<double>("drMax");
  drVetoBarrel_ = config.getParameter<double>("drVetoBarrel");
  drVetoEndcap_ = config.getParameter<double>("drVetoEndcap");
  ptMin_ = config.getParameter<double>("ptMin");
  dzMax_ = config.getParameter<double>("dzMax");
  dxyMax_ = config.getParameter<double>("dxyMax");
  pfToUse_ = config.getParameter<int>("pfCandidateType");

  //register your products
  if(useSCRefs_) 
    produces < reco::RecoEcalCandidateIsolationMap >();
  else 
    produces < reco::ElectronIsolationMap >();
}


EgammaHLTPFChargedIsolationProducer::~EgammaHLTPFChargedIsolationProducer()
{}

void EgammaHLTPFChargedIsolationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pfCandidatesProducer",  edm::InputTag("hltParticleFlowReg"));
  desc.add<edm::InputTag>("recoEcalCandidateProducer", edm::InputTag("hltL1SeededRecoEcalCandidatePF"));
  desc.add<edm::InputTag>("electronProducer", edm::InputTag("hltEle27WP80PixelMatchElectronsL1SeededPF"));
  desc.add<edm::InputTag>("beamSpotProducer", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<bool>("useGsfTrack", false);
  desc.add<bool>("useSCRefs", false);
  desc.add<double>("drMax", 0.3);
  desc.add<double>("drVetoBarrel", 0.02);
  desc.add<double>("drVetoEndcap", 0.02);
  desc.add<double>("ptMin", 0.0);
  desc.add<double>("dzMax", 0.2);
  desc.add<double>("dxyMax", 0.1);
  desc.add<int>("pfCandidateType", 1);
  descriptions.add(("hltEgammaHLTPFChargedIsolationProducer"), desc);
}

void EgammaHLTPFChargedIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<reco::ElectronCollection> electronHandle;
  iEvent.getByLabel(electronProducer_,electronHandle);

 // Get the general tracks
  edm::Handle<reco::PFCandidateCollection> pfHandle;
  iEvent.getByLabel(pfCandidateProducer_, pfHandle);

  reco::ElectronIsolationMap eleMap;
  reco::RecoEcalCandidateIsolationMap recoEcalCandMap;

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
   
  if(useSCRefs_) {
    edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
    iEvent.getByLabel(recoEcalCandidateProducer_, recoEcalCandHandle);

    iEvent.getByLabel(beamSpotProducer_, recoBeamSpotHandle);
    const reco::BeamSpot::Point& beamSpotPosition = recoBeamSpotHandle->position(); 

    for(unsigned int iReco=0; iReco<recoEcalCandHandle->size(); iReco++) {
      
      reco::RecoEcalCandidateRef candRef(recoEcalCandHandle, iReco);
      
      float dRveto = -1;
      if (fabs(candRef->eta())<1.479)
	dRveto = drVetoBarrel_;
      else
	dRveto = drVetoEndcap_;
      
      const reco::PFCandidateCollection* forIsolation = pfHandle.product();
      
      // Shift the photon according to the vertex
      math::XYZVector photon_directionWrtVtx(candRef->superCluster()->x() - beamSpotPosition.x(),
					     candRef->superCluster()->y() - beamSpotPosition.y(),
					     candRef->superCluster()->z() - beamSpotPosition.z());
      
      float sum = 0;
      // Loop over the PFCandidates
      for(unsigned i=0; i<forIsolation->size(); i++) {
	
	const reco::PFCandidate& pfc = (*forIsolation)[i];
	
	//require that PFCandidate is a charged hadron
	if (pfc.particleId() == pfToUse_) {
	  if (pfc.pt() < ptMin_)
	    continue;
        
	  float dz = fabs(pfc.trackRef()->dz(beamSpotPosition));
	  if (dz > dzMax_) continue;
	
	  float dxy = fabs(pfc.trackRef()->dxy(beamSpotPosition));
	  if(fabs(dxy) > dxyMax_) continue;
	
	  float dR = deltaR(photon_directionWrtVtx.Eta(), photon_directionWrtVtx.Phi(), pfc.momentum().Eta(), pfc.momentum().Phi());
	  if(dR > drMax_ || dR < dRveto) continue;
	
	  sum += pfc.pt();
	}
      }
      
      recoEcalCandMap.insert(candRef, sum);
    }
  } else {
    for(unsigned int iEl=0; iEl<electronHandle->size(); iEl++) {
      reco::ElectronRef eleRef(electronHandle, iEl);
      
      //const reco::Track* eleTrk = useGsfTrack_ ? &*eleRef->gsfTrack() : &*eleRef->track();
      float dRveto = -1;
      if (fabs(eleRef->eta())<1.479)
	dRveto = drVetoBarrel_;
      else
	dRveto = drVetoEndcap_;
      
      const reco::PFCandidateCollection* forIsolation = pfHandle.product();
      
      float sum = 0;
      // Loop over the PFCandidates
      for(unsigned i=0; i<forIsolation->size(); i++) {
	
	const reco::PFCandidate& pfc = (*forIsolation)[i];
	
	// FIXME Rimuovi la traccia dell'elettrone esplicitamente
	//require that PFCandidate is a charged hadron
	if (pfc.particleId() == pfToUse_) {
	  if (pfc.pt() < ptMin_)
	    continue;
        
	  float dz = fabs(pfc.trackRef()->dz(eleRef->vertex()));
	  if (dz > dzMax_) continue;
	  
	  float dxy = fabs(pfc.trackRef()->dxy(eleRef->vertex()));
	  if(fabs(dxy) > dxyMax_) continue;
	  
	  float dR = deltaR(eleRef->eta(), eleRef->phi(), pfc.momentum().Eta(), pfc.momentum().Phi());
	  if(dR > drMax_ || dR < dRveto) continue;
	
	  sum += pfc.pt();
	}
      }

      eleMap.insert(eleRef, sum);
    }   
  }

  if(useSCRefs_){
    std::auto_ptr<reco::RecoEcalCandidateIsolationMap> mapForEvent(new reco::RecoEcalCandidateIsolationMap(recoEcalCandMap));
    iEvent.put(mapForEvent);
  }else{
    std::auto_ptr<reco::ElectronIsolationMap> mapForEvent(new reco::ElectronIsolationMap(eleMap));
    iEvent.put(mapForEvent);
  }
}
