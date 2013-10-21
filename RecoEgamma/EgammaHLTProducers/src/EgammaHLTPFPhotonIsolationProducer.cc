/**
 *
 *  \author Matteo Sani (UCSD)
 *
 * $Id:
 */

#include <iostream>
#include <vector>
#include <memory>

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTPFPhotonIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <DataFormats/Math/interface/deltaR.h>

EgammaHLTPFPhotonIsolationProducer::EgammaHLTPFPhotonIsolationProducer(const edm::ParameterSet& config) : conf_(config) {

  // use configuration file to setup input/output collection names
  pfCandidates_           = conf_.getParameter<edm::InputTag>("pfCandidatesProducer");
  recoEcalCandidateProducer_    = conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer");

  drMax_          = conf_.getParameter<double>("drMax");
  drVetoBarrel_   = conf_.getParameter<double>("drVetoBarrel");
  drVetoEndcap_   = conf_.getParameter<double>("drVetoEndcap");
  etaStripBarrel_ = conf_.getParameter<double>("etaStripBarrel");
  etaStripEndcap_ = conf_.getParameter<double>("etaStripEndcap");
  energyBarrel_   = conf_.getParameter<double>("energyBarrel");
  energyEndcap_   = conf_.getParameter<double>("energyEndcap");
  pfToUse_        = conf_.getParameter<int>("pfCandidateType");
  
  produces < reco::RecoEcalCandidateIsolationMap >();
}

EgammaHLTPFPhotonIsolationProducer::~EgammaHLTPFPhotonIsolationProducer()
{}

void EgammaHLTPFPhotonIsolationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pfCandidatesProducer", edm::InputTag("hltParticleFlowReg"));
  desc.add<edm::InputTag>("recoEcalCandidateProducer", edm::InputTag("hltL1SeededRecoEcalCandidatePF"));
  desc.add<double>("drMax", 0.3);
  desc.add<double>("drVetoBarrel", 0.0);
  desc.add<double>("drVetoEndcap", 0.070);
  desc.add<double>("etaStripBarrel", 0.015);
  desc.add<double>("etaStripEndcap", 0.0);
  desc.add<double>("energyBarrel", 0.0);
  desc.add<double>("energyEndcap", 0.0);
  desc.add<int>("pfCandidateType", 4);
  descriptions.add(("hltEgammaHLTPFPhotonIsolationProducer"), desc);
}

void EgammaHLTPFPhotonIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

  edm::Handle<reco::PFCandidateCollection> pfHandle;
  iEvent.getByLabel(pfCandidates_, pfHandle);

  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByLabel(recoEcalCandidateProducer_,recoecalcandHandle);

  reco::RecoEcalCandidateIsolationMap isoMap;

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
      
    const reco::PFCandidateCollection* forIsolation = pfHandle.product();

    float sum = 0;
    for(unsigned i=0; i<forIsolation->size(); i++) {
    
      const reco::PFCandidate& pfc = (*forIsolation)[i];
      
      if (pfc.particleId() ==  pfToUse_) {
	
	// FIXME
	// Do not include the PFCandidate associated by SC Ref to the reco::Photon
	//if(pfc.superClusterRef().isNonnull() && localPho->superCluster().isNonnull()) {
	//  if (pfc.superClusterRef() == localPho->superCluster()) 
	//    continue;
	//}
	
	if (fabs(candRef->eta()) < 1.479) {
	  if (fabs(pfc.pt()) < energyBarrel_)
	    continue;
	} else {
	  if (fabs(pfc.energy()) < energyEndcap_)
	    continue;
	}
	
	// Shift the photon direction vector according to the PF vertex
	math::XYZPoint pfvtx = pfc.vertex();
	math::XYZVector photon_directionWrtVtx(candRef->superCluster()->x() - pfvtx.x(),
					       candRef->superCluster()->y() - pfvtx.y(),
					       candRef->superCluster()->z() - pfvtx.z());
	
	float dEta = fabs(photon_directionWrtVtx.Eta() - pfc.momentum().Eta());
	float dR = deltaR(photon_directionWrtVtx.Eta(), photon_directionWrtVtx.Phi(), pfc.momentum().Eta(), pfc.momentum().Phi());
	
	if (dEta < etaStrip)
	  continue;
	
	if(dR > drMax_ || dR < dRVeto)
	  continue;
	
	sum += pfc.pt();
      }
    }

    isoMap.insert(candRef, sum);
  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);
}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTPFPhotonIsolationProducer);
