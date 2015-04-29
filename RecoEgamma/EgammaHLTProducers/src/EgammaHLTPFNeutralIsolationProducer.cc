/**
 *
 *  \author Matteo Sani (UCSD)
 *
 * $Id:
 */

#include <iostream>
#include <vector>
#include <memory>

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTPFNeutralIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include <DataFormats/Math/interface/deltaR.h>

EgammaHLTPFNeutralIsolationProducer::EgammaHLTPFNeutralIsolationProducer(const edm::ParameterSet& config) {

  pfCandidateProducer_       = consumes<reco::PFCandidateCollection>(config.getParameter<edm::InputTag>("pfCandidatesProducer"));

  useSCRefs_ = config.getParameter<bool>("useSCRefs");

  drMax_          = config.getParameter<double>("drMax");
  drVetoBarrel_   = config.getParameter<double>("drVetoBarrel");
  drVetoEndcap_   = config.getParameter<double>("drVetoEndcap");
  etaStripBarrel_ = config.getParameter<double>("etaStripBarrel");
  etaStripEndcap_ = config.getParameter<double>("etaStripEndcap");
  energyBarrel_   = config.getParameter<double>("energyBarrel");
  energyEndcap_   = config.getParameter<double>("energyEndcap");
  pfToUse_        = config.getParameter<int>("pfCandidateType");

  doRhoCorrection_                = config.getParameter<bool>("doRhoCorrection");
  if (doRhoCorrection_)
    rhoProducer_                    = consumes<double>(config.getParameter<edm::InputTag>("rhoProducer"));
  
  rhoMax_                         = config.getParameter<double>("rhoMax"); 
  rhoScale_                       = config.getParameter<double>("rhoScale"); 
  effectiveAreaBarrel_            = config.getParameter<double>("effectiveAreaBarrel");
  effectiveAreaEndcap_            = config.getParameter<double>("effectiveAreaEndcap");

  if(useSCRefs_) {
    produces < reco::RecoEcalCandidateIsolationMap >();
    recoEcalCandidateProducer_ = consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"));
  } else {
    produces < reco::ElectronIsolationMap >();
    electronProducer_          = consumes<reco::ElectronCollection>(config.getParameter<edm::InputTag>("electronProducer"));
  }
}

void EgammaHLTPFNeutralIsolationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("electronProducer", edm::InputTag("hltEle27WP80PixelMatchElectronsL1SeededPF"));
  desc.add<edm::InputTag>("recoEcalCandidateProducer", edm::InputTag("hltL1SeededRecoEcalCandidatePF"));
  desc.add<edm::InputTag>("pfCandidatesProducer",  edm::InputTag("hltParticleFlowReg"));
  desc.add<edm::InputTag>("rhoProducer", edm::InputTag("fixedGridRhoFastjetAllCalo"));
  desc.add<bool>("doRhoCorrection", false);
  desc.add<double>("rhoMax", 9.9999999E7); 
  desc.add<double>("rhoScale", 1.0); 
  desc.add<double>("effectiveAreaBarrel", 0.101);
  desc.add<double>("effectiveAreaEndcap", 0.046);
  desc.add<bool>("useSCRefs", false);
  desc.add<double>("drMax", 0.3);
  desc.add<double>("drVetoBarrel", 0.0);
  desc.add<double>("drVetoEndcap", 0.0);
  desc.add<double>("etaStripBarrel", 0.0);
  desc.add<double>("etaStripEndcap", 0.0);
  desc.add<double>("energyBarrel", 0.0);
  desc.add<double>("energyEndcap", 0.0);
  desc.add<int>("pfCandidateType", 5);
  descriptions.add(("hltEgammaHLTPFNeutralIsolationProducer"), desc);
}

void EgammaHLTPFNeutralIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

  edm::Handle<double> rhoHandle;
  double rho = 0.0;
  if (doRhoCorrection_) {
    iEvent.getByToken(rhoProducer_, rhoHandle);
    rho = *(rhoHandle.product());
  }
  
  if (rho > rhoMax_)
    rho = rhoMax_;
  
  rho = rho*rhoScale_;

  edm::Handle<reco::ElectronCollection> electronHandle;
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  edm::Handle<reco::PFCandidateCollection> pfHandle;

  iEvent.getByToken(pfCandidateProducer_, pfHandle);
  const reco::PFCandidateCollection* forIsolation = pfHandle.product();

  if(useSCRefs_) {

    iEvent.getByToken(recoEcalCandidateProducer_,recoecalcandHandle);
    reco::RecoEcalCandidateIsolationMap recoEcalCandMap(recoecalcandHandle);

    float dRVeto = -1.;
    float etaStrip = -1;
    
    for (unsigned int iReco = 0; iReco < recoecalcandHandle->size(); iReco++) {
      reco::RecoEcalCandidateRef candRef(recoecalcandHandle, iReco);
      
      if (fabs(candRef->eta()) < 1.479) {
	dRVeto = drVetoBarrel_;
	etaStrip = etaStripBarrel_;
      } else {
	dRVeto = drVetoEndcap_;
	etaStrip = etaStripEndcap_;
      }
      
      float sum = 0;

      // Loop over the PFCandidates
      for(unsigned i=0; i<forIsolation->size(); i++) {
	const reco::PFCandidate& pfc = (*forIsolation)[i];
	
	//require that the PFCandidate is a neutral hadron
	if (pfc.particleId() ==  pfToUse_) {
	  
	  if (fabs(candRef->eta()) < 1.479) {
	    if (fabs(pfc.pt()) < energyBarrel_)
	      continue;
	  } else {
	    if (fabs(pfc.energy()) < energyEndcap_)
	      continue;
	  }
	  
	  // Shift the RecoEcalCandidate direction vector according to the PF vertex
	  math::XYZPoint pfvtx = pfc.vertex();
	  math::XYZVector candDirectionWrtVtx(candRef->superCluster()->x() - pfvtx.x(),
					      candRef->superCluster()->y() - pfvtx.y(),
					      candRef->superCluster()->z() - pfvtx.z());
	  
	  float dEta = fabs(candDirectionWrtVtx.Eta() - pfc.momentum().Eta());
	  if(dEta < etaStrip) continue;
	  
	  float dR = deltaR(candDirectionWrtVtx.Eta(), candDirectionWrtVtx.Phi(), pfc.momentum().Eta(), pfc.momentum().Phi());
	  if(dR > drMax_ || dR < dRVeto) continue;
	  
	  sum += pfc.pt();
	}
      }
          
      if (doRhoCorrection_) {
	if (fabs(candRef->eta()) < 1.479) 
	  sum = sum - rho*effectiveAreaBarrel_;
	else
	  sum = sum - rho*effectiveAreaEndcap_;
      }
       
      recoEcalCandMap.insert(candRef, sum);
    }
    std::auto_ptr<reco::RecoEcalCandidateIsolationMap> mapForEvent(new reco::RecoEcalCandidateIsolationMap(recoEcalCandMap));
    iEvent.put(mapForEvent);
    
  } else {

    iEvent.getByToken(electronProducer_,electronHandle);
    reco::ElectronIsolationMap eleMap(electronHandle);
    
    float dRVeto = -1.;
    float etaStrip = -1;

    for(unsigned int iEl=0; iEl<electronHandle->size(); iEl++) {
      reco::ElectronRef eleRef(electronHandle, iEl);
      
      if (fabs(eleRef->eta()) < 1.479) {
	dRVeto = drVetoBarrel_;
	etaStrip = etaStripBarrel_;
      } else {
	dRVeto = drVetoEndcap_;
	etaStrip = etaStripEndcap_;
      }
      
      float sum = 0;

      // Loop over the PFCandidates
      for(unsigned i=0; i<forIsolation->size(); i++) {
	const reco::PFCandidate& pfc = (*forIsolation)[i];
	
	//require that the PFCandidate is a neutral hadron
	if (pfc.particleId() ==  pfToUse_) {
	  
	  if (fabs(eleRef->eta()) < 1.479) {
	    if (fabs(pfc.pt()) < energyBarrel_)
	      continue;
	  } else {
	    if (fabs(pfc.energy()) < energyEndcap_)
	      continue;
	  }
	  
	  float dEta = fabs(eleRef->eta() - pfc.momentum().Eta());
	  if(dEta < etaStrip) continue;
	  
	  float dR = deltaR(eleRef->eta(), eleRef->phi(), pfc.momentum().Eta(), pfc.momentum().Phi());
	  if(dR > drMax_ || dR < dRVeto) continue;
	  
	  sum += pfc.pt();
	}
      }
    
      if (doRhoCorrection_) {
	if (fabs(eleRef->superCluster()->eta()) < 1.479) 
	  sum = sum - rho*effectiveAreaBarrel_;
	else
	  sum = sum - rho*effectiveAreaEndcap_;
      }
 
      eleMap.insert(eleRef, sum);
    }   
    std::auto_ptr<reco::ElectronIsolationMap> mapForEvent(new reco::ElectronIsolationMap(eleMap));
    iEvent.put(mapForEvent);
  }
}
