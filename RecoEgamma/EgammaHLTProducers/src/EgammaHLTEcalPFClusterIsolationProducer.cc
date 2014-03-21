#include <iostream>
#include <vector>
#include <memory>

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTEcalPFClusterIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include <DataFormats/Math/interface/deltaR.h>

EgammaHLTEcalPFClusterIsolationProducer::EgammaHLTEcalPFClusterIsolationProducer(const edm::ParameterSet& config) {

  recoEcalCandidateProducer_ = consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"));
  pfClusterProducer_         = consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducer"));

  drMax_          = config.getParameter<double>("drMax");
  drVetoBarrel_   = config.getParameter<double>("drVetoBarrel");
  drVetoEndcap_   = config.getParameter<double>("drVetoEndcap");
  etaStripBarrel_ = config.getParameter<double>("etaStripBarrel");
  etaStripEndcap_ = config.getParameter<double>("etaStripEndcap");
  energyBarrel_   = config.getParameter<double>("energyBarrel");
  energyEndcap_   = config.getParameter<double>("energyEndcap");
  
  produces < reco::RecoEcalCandidateIsolationMap >();

}

EgammaHLTEcalPFClusterIsolationProducer::~EgammaHLTEcalPFClusterIsolationProducer()
{}

void EgammaHLTEcalPFClusterIsolationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recoEcalCandidateProducer", edm::InputTag("hltL1SeededRecoEcalCandidatePF"));
  desc.add<edm::InputTag>("pfClusterProducer", edm::InputTag("hltParticleFlowClusterECAL"));
  desc.add<double>("drMax", 0.3);
  desc.add<double>("drVetoBarrel", 0.0);
  desc.add<double>("drVetoEndcap", 0.070);
  desc.add<double>("etaStripBarrel", 0.015);
  desc.add<double>("etaStripEndcap", 0.0);
  desc.add<double>("energyBarrel", 0.0);
  desc.add<double>("energyEndcap", 0.0);
  descriptions.add(("hltEgammaHLTEcalPFClusterIsolationProducer"), desc);
}

void EgammaHLTEcalPFClusterIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  edm::Handle<reco::PFClusterCollection> clusterHandle;

  iEvent.getByToken(recoEcalCandidateProducer_,recoecalcandHandle);
  iEvent.getByToken(pfClusterProducer_, clusterHandle);
  const reco::PFClusterCollection* forIsolation = clusterHandle.product();

  reco::RecoEcalCandidateIsolationMap recoEcalCandMap;
  
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
    
    // Loop over the PFClusters
    for(unsigned i=0; i<forIsolation->size(); i++) {
      const reco::PFCluster& pfclu = (*forIsolation)[i];
      
      if (fabs(candRef->eta()) < 1.479) {
	if (fabs(pfclu.pt()) < energyBarrel_)
	  continue;
      } else {
	if (fabs(pfclu.energy()) < energyEndcap_)
	  continue;
      }

      float dEta = fabs(candRef->eta() - pfclu.eta());
      if(dEta < etaStrip) continue;
      
      float dR = deltaR(candRef->eta(), candRef->phi(), pfclu.eta(), pfclu.phi());
      if(dR > drMax_ || dR < dRVeto) continue;
      
      sum += pfclu.pt();
    }
    
    recoEcalCandMap.insert(candRef, sum);
  }
  
  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> mapForEvent(new reco::RecoEcalCandidateIsolationMap(recoEcalCandMap));
  iEvent.put(mapForEvent);

}
