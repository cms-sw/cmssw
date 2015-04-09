#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaEcalPFClusterIsolationProducer.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EcalPFClusterIsolation.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

template<typename T1>
EgammaEcalPFClusterIsolationProducer<T1>::EgammaEcalPFClusterIsolationProducer(const edm::ParameterSet& config): 

  emObjectProducer_   (consumes<T1Collection>(config.getParameter<edm::InputTag>("candidateProducer"))),
  pfClusterProducer_  (consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducer"))),
  drMax_              (config.getParameter<double>("drMax")),
  drVetoBarrel_       (config.getParameter<double>("drVetoBarrel")),
  drVetoEndcap_       (config.getParameter<double>("drVetoEndcap")),
  etaStripBarrel_     (config.getParameter<double>("etaStripBarrel")),
  etaStripEndcap_     (config.getParameter<double>("etaStripEndcap")),
  energyBarrel_       (config.getParameter<double>("energyBarrel")),
  energyEndcap_       (config.getParameter<double>("energyEndcap")) {

  produces <edm::ValueMap<float>>();
}

template<typename T1>
EgammaEcalPFClusterIsolationProducer<T1>::~EgammaEcalPFClusterIsolationProducer()
{}

template<typename T1>
void EgammaEcalPFClusterIsolationProducer<T1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("candidateProducer", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("pfClusterProducer", edm::InputTag("particleFlowClusterECAL")); 
  desc.add<double>("drMax", 0.3);
  desc.add<double>("drVetoBarrel", 0.0);
  desc.add<double>("drVetoEndcap", 0.0);
  desc.add<double>("etaStripBarrel", 0.0);
  desc.add<double>("etaStripEndcap", 0.0);
  desc.add<double>("energyBarrel", 0.0);
  desc.add<double>("energyEndcap", 0.0);
  descriptions.add(defaultModuleLabel<EgammaEcalPFClusterIsolationProducer<T1>>(), desc);
}

template<typename T1>
void EgammaEcalPFClusterIsolationProducer<T1>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<T1Collection> emObjectHandle;
  iEvent.getByToken(emObjectProducer_, emObjectHandle);

  std::auto_ptr<edm::ValueMap<float> > isoMap(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler(*isoMap);
  std::vector<float> retV(emObjectHandle->size(),0);

  edm::Handle<reco::PFClusterCollection> clusterHandle;
  iEvent.getByToken(pfClusterProducer_, clusterHandle);

  EcalPFClusterIsolation<T1> isoAlgo(drMax_, drVetoBarrel_, drVetoEndcap_, etaStripBarrel_, etaStripEndcap_, energyBarrel_, energyEndcap_);
  
  for (unsigned int iReco = 0; iReco < emObjectHandle->size(); iReco++) {
    T1Ref candRef(emObjectHandle, iReco);
    retV[iReco] = isoAlgo.getSum(candRef, clusterHandle);
  }
  
  filler.insert(emObjectHandle,retV.begin(),retV.end());
  filler.fill();

  iEvent.put(isoMap);
}

typedef EgammaEcalPFClusterIsolationProducer<reco::GsfElectron> ElectronEcalPFClusterIsolationProducer;
typedef EgammaEcalPFClusterIsolationProducer<reco::Photon> PhotonEcalPFClusterIsolationProducer;

DEFINE_FWK_MODULE(ElectronEcalPFClusterIsolationProducer);
DEFINE_FWK_MODULE(PhotonEcalPFClusterIsolationProducer);
