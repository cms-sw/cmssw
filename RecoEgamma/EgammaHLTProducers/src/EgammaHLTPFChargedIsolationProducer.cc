/** 
 *
 *  \author Matteo Sani (UCSD)
 * 
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTPFChargedIsolationProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"

#include <DataFormats/Math/interface/deltaR.h>

EgammaHLTPFChargedIsolationProducer::EgammaHLTPFChargedIsolationProducer(const edm::ParameterSet& config):
  pfCandidateProducer_(consumes<reco::PFCandidateCollection>(config.getParameter<edm::InputTag>("pfCandidatesProducer"))),
  beamSpotProducer_   (consumes<reco::BeamSpot>(config.getParameter<edm::InputTag>("beamSpotProducer"))),
  useGsfTrack_        (config.getParameter<bool>("useGsfTrack")),
  useSCRefs_          (config.getParameter<bool>("useSCRefs")),
  drMax_              (config.getParameter<double>("drMax")),
  drVetoBarrel_       (config.getParameter<double>("drVetoBarrel")),
  drVetoEndcap_       (config.getParameter<double>("drVetoEndcap")),
  ptMin_              (config.getParameter<double>("ptMin")),
  dzMax_              (config.getParameter<double>("dzMax")),
  dxyMax_             (config.getParameter<double>("dxyMax")),
  pfToUse_            (config.getParameter<int>("pfCandidateType")) {

  if(useSCRefs_) {
    recoEcalCandidateProducer_ = consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"));
    produces < reco::RecoEcalCandidateIsolationMap >();
  } else {
    electronProducer_          = consumes<reco::ElectronCollection>(config.getParameter<edm::InputTag>("electronProducer"));
    produces < reco::ElectronIsolationMap >();
  }
}

void EgammaHLTPFChargedIsolationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("electronProducer", edm::InputTag("hltEle27WP80PixelMatchElectronsL1SeededPF"));
  desc.add<edm::InputTag>("recoEcalCandidateProducer", edm::InputTag("hltL1SeededRecoEcalCandidatePF"));
  desc.add<edm::InputTag>("pfCandidatesProducer",  edm::InputTag("hltParticleFlowReg"));
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
  edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
  edm::Handle<reco::PFCandidateCollection> pfHandle;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;

  iEvent.getByToken(pfCandidateProducer_, pfHandle);
  const reco::PFCandidateCollection* forIsolation = pfHandle.product();

  if(useSCRefs_) {

    iEvent.getByToken(recoEcalCandidateProducer_, recoEcalCandHandle);
    reco::RecoEcalCandidateIsolationMap recoEcalCandMap(recoEcalCandHandle);

    iEvent.getByToken(beamSpotProducer_, recoBeamSpotHandle);
    const reco::BeamSpot::Point& beamSpotPosition = recoBeamSpotHandle->position(); 

    float dRveto = -1;

    for(unsigned int iReco=0; iReco<recoEcalCandHandle->size(); iReco++) {
      reco::RecoEcalCandidateRef candRef(recoEcalCandHandle, iReco);
      
      if (fabs(candRef->eta())<1.479)
	dRveto = drVetoBarrel_;
      else
	dRveto = drVetoEndcap_;
      
      // Shift the RecoEcalCandidate direction vector according to the vertex
      math::XYZVector candDirectionWrtVtx(candRef->superCluster()->x() - beamSpotPosition.x(),
					  candRef->superCluster()->y() - beamSpotPosition.y(),
					  candRef->superCluster()->z() - beamSpotPosition.z());
      
      float sum = 0;

      // Loop over the PFCandidates
      for(unsigned i=0; i<forIsolation->size(); i++) {
	const reco::PFCandidate& pfc = (*forIsolation)[i];
	
	//require that the PFCandidate is a charged hadron
	if (pfc.particleId() == pfToUse_) {

	  if(pfc.pt() < ptMin_) continue;
        
	  float dz = fabs(pfc.trackRef()->dz(beamSpotPosition));
	  if(dz > dzMax_) continue;
	
	  float dxy = fabs(pfc.trackRef()->dxy(beamSpotPosition));
	  if(fabs(dxy) > dxyMax_) continue;
	
	  float dR = deltaR(candDirectionWrtVtx.Eta(), candDirectionWrtVtx.Phi(), pfc.momentum().Eta(), pfc.momentum().Phi());
	  if(dR > drMax_ || dR < dRveto) continue;
	
	  sum += pfc.pt();
	}
      }
      
      recoEcalCandMap.insert(candRef, sum);
    }
    std::auto_ptr<reco::RecoEcalCandidateIsolationMap> mapForEvent(new reco::RecoEcalCandidateIsolationMap(recoEcalCandMap));
    iEvent.put(mapForEvent);

  } else {

    iEvent.getByToken(electronProducer_,electronHandle);
    reco::ElectronIsolationMap eleMap(electronHandle);   

    float dRveto = -1;

    for(unsigned int iEl=0; iEl<electronHandle->size(); iEl++) {
      reco::ElectronRef eleRef(electronHandle, iEl);
      //const reco::Track* eleTrk = useGsfTrack_ ? &*eleRef->gsfTrack() : &*eleRef->track();

      if (fabs(eleRef->eta())<1.479)
	dRveto = drVetoBarrel_;
      else
	dRveto = drVetoEndcap_;
      
      float sum = 0;

      // Loop over the PFCandidates
      for(unsigned i=0; i<forIsolation->size(); i++) {
	const reco::PFCandidate& pfc = (*forIsolation)[i];
	
	//require that the PFCandidate is a charged hadron
	if (pfc.particleId() == pfToUse_) {

	  if(pfc.pt() < ptMin_) continue;
        
 	  float dz = fabs(pfc.trackRef()->dz(eleRef->vertex()));
 	  if(dz > dzMax_) continue;
	  
 	  float dxy = fabs(pfc.trackRef()->dxy(eleRef->vertex()));
 	  if(fabs(dxy) > dxyMax_) continue;
	  
	  float dR = deltaR(eleRef->eta(), eleRef->phi(), pfc.momentum().Eta(), pfc.momentum().Phi());
	  if(dR > drMax_ || dR < dRveto) continue;
	
	  sum += pfc.pt();
	}
      }

      eleMap.insert(eleRef, sum);
    }   
    std::auto_ptr<reco::ElectronIsolationMap> mapForEvent(new reco::ElectronIsolationMap(eleMap));
    iEvent.put(mapForEvent);
  }
}
