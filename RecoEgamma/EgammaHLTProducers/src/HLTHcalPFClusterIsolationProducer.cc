#include <iostream>
#include <vector>
#include <memory>

#include "RecoEgamma/EgammaHLTProducers/interface/HLTHcalPFClusterIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <DataFormats/Math/interface/deltaR.h>

template<typename T1>
HLTHcalPFClusterIsolationProducer<T1>::HLTHcalPFClusterIsolationProducer(const edm::ParameterSet& config) :
  pfClusterProducerHCAL_  ( consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducerHCAL"))),
  rhoProducer_            ( consumes<double>(config.getParameter<edm::InputTag>("rhoProducer"))),
  pfClusterProducerHFEM_  ( consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducerHFEM"))),
  pfClusterProducerHFHAD_ ( consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducerHFHAD"))),
  useHF_                  ( config.getParameter<bool>("useHF")),
  drMax_                  ( config.getParameter<double>("drMax")),
  drVetoBarrel_           ( config.getParameter<double>("drVetoBarrel")),
  drVetoEndcap_           ( config.getParameter<double>("drVetoEndcap")),
  etaStripBarrel_         ( config.getParameter<double>("etaStripBarrel")),
  etaStripEndcap_         ( config.getParameter<double>("etaStripEndcap")),
  energyBarrel_           ( config.getParameter<double>("energyBarrel")),
  energyEndcap_           ( config.getParameter<double>("energyEndcap")),
  doRhoCorrection_        ( config.getParameter<bool>("doRhoCorrection")),
  rhoMax_                 ( config.getParameter<double>("rhoMax")),
  rhoScale_               ( config.getParameter<double>("rhoScale")), 
  effectiveAreaBarrel_    ( config.getParameter<double>("effectiveAreaBarrel")),
  effectiveAreaEndcap_    ( config.getParameter<double>("effectiveAreaEndcap")) {
  
  std::string recoCandidateProducerName = "recoCandidateProducer";
  if ((typeid(HLTHcalPFClusterIsolationProducer<T1>) == typeid(HLTHcalPFClusterIsolationProducer<reco::RecoEcalCandidate>))) recoCandidateProducerName = "recoEcalCandidateProducer";
  recoCandidateProducer_ = consumes<T1Collection>(config.getParameter<edm::InputTag>(recoCandidateProducerName));
  
  produces <T1IsolationMap >();
}

template<typename T1>
HLTHcalPFClusterIsolationProducer<T1>::~HLTHcalPFClusterIsolationProducer()
{}

template<typename T1>
void HLTHcalPFClusterIsolationProducer<T1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
 
  std::string recoCandidateProducerName = "recoCandidateProducer";
  if ((typeid(HLTHcalPFClusterIsolationProducer<T1>) == typeid(HLTHcalPFClusterIsolationProducer<reco::RecoEcalCandidate>))) recoCandidateProducerName = "recoEcalCandidateProducer";

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(recoCandidateProducerName, edm::InputTag("hltL1SeededRecoEcalCandidatePF"));
  desc.add<edm::InputTag>("pfClusterProducerHCAL", edm::InputTag("hltParticleFlowClusterHCAL"));
  desc.ifValue(edm::ParameterDescription<bool>("useHF", false, true),
	       true >> (edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFEM", edm::InputTag("hltParticleFlowClusterHFEM"), true) and
			edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFHAD", edm::InputTag("hltParticleFlowClusterHFHAD"), true)) or
	       false >> (edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFEM", edm::InputTag(""), true) and
			 edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFHAD", edm::InputTag(""), true)));
  desc.add<edm::InputTag>("rhoProducer", edm::InputTag("fixedGridRhoFastjetAllCalo"));
  desc.add<bool>("doRhoCorrection", false);
  desc.add<double>("rhoMax", 9.9999999E7); 
  desc.add<double>("rhoScale", 1.0); 
  desc.add<double>("effectiveAreaBarrel", 0.101);
  desc.add<double>("effectiveAreaEndcap", 0.046);
  desc.add<double>("drMax", 0.3);
  desc.add<double>("drVetoBarrel", 0.0);
  desc.add<double>("drVetoEndcap", 0.0);
  desc.add<double>("etaStripBarrel", 0.0);
  desc.add<double>("etaStripEndcap", 0.0);
  desc.add<double>("energyBarrel", 0.0);
  desc.add<double>("energyEndcap", 0.0);
  descriptions.add(std::string("hlt")+std::string(typeid(HLTHcalPFClusterIsolationProducer<T1>).name()), desc);
}

template<typename T1>
void HLTHcalPFClusterIsolationProducer<T1>::produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  
  edm::Handle<double> rhoHandle;
  double rho = 0.0;
  if (doRhoCorrection_) {
    iEvent.getByToken(rhoProducer_, rhoHandle);
    rho = *(rhoHandle.product());
  }
  
  if (rho > rhoMax_)
    rho = rhoMax_;
  
  rho = rho*rhoScale_;

  edm::Handle<T1Collection> recoCandHandle;
  edm::Handle<reco::PFClusterCollection> clusterHcalHandle;
  edm::Handle<reco::PFClusterCollection> clusterHfemHandle;
  edm::Handle<reco::PFClusterCollection> clusterHfhadHandle;

  iEvent.getByToken(recoCandidateProducer_,recoCandHandle);
  iEvent.getByToken(pfClusterProducerHCAL_, clusterHcalHandle);
  const reco::PFClusterCollection* forIsolationHcal = clusterHcalHandle.product();

  if (useHF_) {
    iEvent.getByToken(pfClusterProducerHFEM_, clusterHfemHandle);
    iEvent.getByToken(pfClusterProducerHFHAD_, clusterHfhadHandle);
  }

  T1IsolationMap recoCandMap;
  
  float dRVeto = -1.;
  float etaStrip = -1;
  
  for (unsigned int iReco = 0; iReco < recoCandHandle->size(); iReco++) {
    T1Ref candRef(recoCandHandle, iReco);
    
    if (fabs(candRef->eta()) < 1.479) {
      dRVeto = drVetoBarrel_;
      etaStrip = etaStripBarrel_;
    } else {
      dRVeto = drVetoEndcap_;
      etaStrip = etaStripEndcap_;
    }
    
    float sum = 0;
    
    // Loop over the 3 types of PFClusters

    for(unsigned i=0; i<forIsolationHcal->size(); i++) {
      const reco::PFCluster& pfclu = (*forIsolationHcal)[i];
      
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

    if (useHF_) {
      const reco::PFClusterCollection* forIsolationHfem = clusterHfemHandle.product();
      const reco::PFClusterCollection* forIsolationHfhad = clusterHfhadHandle.product();
      
      for(unsigned i=0; i<forIsolationHfem->size(); i++) {
	const reco::PFCluster& pfclu = (*forIsolationHfem)[i];
	
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
      
      for(unsigned i=0; i<forIsolationHfhad->size(); i++) {
	const reco::PFCluster& pfclu = (*forIsolationHfhad)[i];
	
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
    }
 
    if (doRhoCorrection_) {
      if (fabs(candRef->eta()) < 1.479) 
	sum = sum - rho*effectiveAreaBarrel_;
      else
	sum = sum - rho*effectiveAreaEndcap_;
    }

    recoCandMap.insert(candRef, sum);
  }
  
  std::auto_ptr<T1IsolationMap> mapForEvent(new T1IsolationMap(recoCandMap));
  iEvent.put(mapForEvent);
}

typedef HLTHcalPFClusterIsolationProducer<reco::RecoEcalCandidate> EgammaHLTHcalPFClusterIsolationProducer;
typedef HLTHcalPFClusterIsolationProducer<reco::RecoChargedCandidate> MuonHLTHcalPFClusterIsolationProducer;

DEFINE_FWK_MODULE(EgammaHLTHcalPFClusterIsolationProducer);
DEFINE_FWK_MODULE(MuonHLTHcalPFClusterIsolationProducer);
