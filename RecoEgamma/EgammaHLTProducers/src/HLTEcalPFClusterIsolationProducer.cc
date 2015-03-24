#include <iostream>
#include <vector>
#include <memory>

#include "RecoEgamma/EgammaHLTProducers/interface/HLTEcalPFClusterIsolationProducer.h"

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

#include "RecoEgamma/EgammaIsolationAlgos/interface/EcalPFClusterIsolation.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template<typename T1>
HLTEcalPFClusterIsolationProducer<T1>::HLTEcalPFClusterIsolationProducer(const edm::ParameterSet& config):
  pfClusterProducer_  (consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducer"))),
  rhoProducer_        (consumes<double>(config.getParameter<edm::InputTag>("rhoProducer"))),
  drMax_              (config.getParameter<double>("drMax")),
  drVetoBarrel_       (config.getParameter<double>("drVetoBarrel")),
  drVetoEndcap_       (config.getParameter<double>("drVetoEndcap")),
  etaStripBarrel_     (config.getParameter<double>("etaStripBarrel")),
  etaStripEndcap_     (config.getParameter<double>("etaStripEndcap")),
  energyBarrel_       (config.getParameter<double>("energyBarrel")),
  energyEndcap_       (config.getParameter<double>("energyEndcap")),
  doRhoCorrection_    (config.getParameter<bool>("doRhoCorrection")),
  rhoMax_             (config.getParameter<double>("rhoMax")),
  rhoScale_           (config.getParameter<double>("rhoScale")),
  effectiveAreaBarrel_(config.getParameter<double>("effectiveAreaBarrel")),
  effectiveAreaEndcap_(config.getParameter<double>("effectiveAreaEndcap")) {

  std::string recoCandidateProducerName = "recoCandidateProducer";
  if ((typeid(HLTEcalPFClusterIsolationProducer<T1>) == typeid(HLTEcalPFClusterIsolationProducer<reco::RecoEcalCandidate>))) recoCandidateProducerName = "recoEcalCandidateProducer";
    
  recoCandidateProducer_ = consumes<T1Collection>(config.getParameter<edm::InputTag>(recoCandidateProducerName));
  produces <T1IsolationMap>();

}

template<typename T1>
HLTEcalPFClusterIsolationProducer<T1>::~HLTEcalPFClusterIsolationProducer()
{}

template<typename T1>
void HLTEcalPFClusterIsolationProducer<T1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  std::string recoCandidateProducerName = "recoCandidateProducer";
  if ((typeid(HLTEcalPFClusterIsolationProducer<T1>) == typeid(HLTEcalPFClusterIsolationProducer<reco::RecoEcalCandidate>))) recoCandidateProducerName = "recoEcalCandidateProducer";
  
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(recoCandidateProducerName, edm::InputTag("hltL1SeededRecoEcalCandidatePF"));
  desc.add<edm::InputTag>("pfClusterProducer", edm::InputTag("hltParticleFlowClusterECAL")); 
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
  descriptions.add(defaultModuleLabel<HLTEcalPFClusterIsolationProducer<T1>>(), desc);
}

template<typename T1>
void HLTEcalPFClusterIsolationProducer<T1>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

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
  edm::Handle<reco::PFClusterCollection> clusterHandle;

  iEvent.getByToken(recoCandidateProducer_,recoCandHandle);
  iEvent.getByToken(pfClusterProducer_, clusterHandle);

  EcalPFClusterIsolation<T1> isoAlgo(drMax_, drVetoBarrel_, drVetoEndcap_, etaStripBarrel_, etaStripEndcap_, energyBarrel_, energyEndcap_);
  T1IsolationMap recoCandMap;
    
  for (unsigned int iReco = 0; iReco < recoCandHandle->size(); iReco++) {
    T1Ref candRef(recoCandHandle, iReco);
    
    float sum = isoAlgo.getSum(candRef, clusterHandle);
    
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

typedef HLTEcalPFClusterIsolationProducer<reco::RecoEcalCandidate> EgammaHLTEcalPFClusterIsolationProducer;
typedef HLTEcalPFClusterIsolationProducer<reco::RecoChargedCandidate> MuonHLTEcalPFClusterIsolationProducer;

DEFINE_FWK_MODULE(EgammaHLTEcalPFClusterIsolationProducer);
DEFINE_FWK_MODULE(MuonHLTEcalPFClusterIsolationProducer);
