#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaHcalPFClusterIsolationProducer.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include "FWCore/Utilities/interface/TypeID.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/HcalPFClusterIsolation.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <typeinfo> 

template<typename T1>
EgammaHcalPFClusterIsolationProducer<T1>::EgammaHcalPFClusterIsolationProducer(const edm::ParameterSet& config): 

  emObjectProducer_       (consumes<T1Collection>(config.getParameter<edm::InputTag>("candidateProducer"))),
  pfClusterProducerHCAL_  (consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducerHCAL"))),
  pfClusterProducerHFEM_  (consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducerHFEM"))),
  pfClusterProducerHFHAD_ (consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducerHFHAD"))),
  useHF_                  (config.getParameter<bool>("useHF")),
  drMax_                  (config.getParameter<double>("drMax")),
  drVetoBarrel_           (config.getParameter<double>("drVetoBarrel")),
  drVetoEndcap_           (config.getParameter<double>("drVetoEndcap")),
  etaStripBarrel_         (config.getParameter<double>("etaStripBarrel")),
  etaStripEndcap_         (config.getParameter<double>("etaStripEndcap")),
  energyBarrel_           (config.getParameter<double>("energyBarrel")),
  energyEndcap_           (config.getParameter<double>("energyEndcap")),
  useEt_                  (config.getParameter<bool>("useEt")) {

  produces <edm::ValueMap<float>>();
}

template<typename T1>
EgammaHcalPFClusterIsolationProducer<T1>::~EgammaHcalPFClusterIsolationProducer()
{}

template<typename T1>
void EgammaHcalPFClusterIsolationProducer<T1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("candidateProducer", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("pfClusterProducerHCAL", edm::InputTag("particleFlowClusterHCAL")); 
  desc.ifValue(edm::ParameterDescription<bool>("useHF", false, true),
               true >> (edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFEM", edm::InputTag("hltParticleFlowClusterHFEM"), true) and
                        edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFHAD", edm::InputTag("hltParticleFlowClusterHFHAD"), true)) or
               false >> (edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFEM", edm::InputTag(""), true) and
                         edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFHAD", edm::InputTag(""), true)));
  desc.add<double>("drMax", 0.3);
  desc.add<double>("drVetoBarrel", 0.0);
  desc.add<double>("drVetoEndcap", 0.0);
  desc.add<double>("etaStripBarrel", 0.0);
  desc.add<double>("etaStripEndcap", 0.0);
  desc.add<double>("energyBarrel", 0.0);
  desc.add<double>("energyEndcap", 0.0);
  desc.add<bool>("useEt", true);
  descriptions.add(defaultModuleLabel<EgammaHcalPFClusterIsolationProducer<T1>>(), desc);
}

template<typename T1>
void EgammaHcalPFClusterIsolationProducer<T1>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<T1Collection> emObjectHandle;
  iEvent.getByToken(emObjectProducer_, emObjectHandle);

  std::auto_ptr<edm::ValueMap<float> > isoMap(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler(*isoMap);
  std::vector<float> retV(emObjectHandle->size(),0);

  std::vector<edm::Handle<reco::PFClusterCollection>> clusterHandles;
  edm::Handle<reco::PFClusterCollection> clusterHandle;
  iEvent.getByToken(pfClusterProducerHCAL_, clusterHandle);
  clusterHandles.push_back(clusterHandle);

  if (useHF_) {
    edm::Handle<reco::PFClusterCollection> clusterHandle;
    iEvent.getByToken(pfClusterProducerHFEM_, clusterHandle);
    clusterHandles.push_back(clusterHandle);

    iEvent.getByToken(pfClusterProducerHFHAD_, clusterHandle);
    clusterHandles.push_back(clusterHandle);
  }

  HcalPFClusterIsolation<T1> isoAlgo(drMax_, drVetoBarrel_, drVetoEndcap_, etaStripBarrel_, etaStripEndcap_, energyBarrel_, energyEndcap_, useEt_);
  
  for (unsigned int iReco = 0; iReco < emObjectHandle->size(); iReco++) {
    T1Ref candRef(emObjectHandle, iReco);
    retV[iReco] = isoAlgo.getSum(candRef, clusterHandles);
  }
  
  filler.insert(emObjectHandle,retV.begin(),retV.end());
  filler.fill();

  iEvent.put(isoMap);
}

typedef EgammaHcalPFClusterIsolationProducer<reco::GsfElectron> ElectronHcalPFClusterIsolationProducer;
typedef EgammaHcalPFClusterIsolationProducer<reco::Photon> PhotonHcalPFClusterIsolationProducer;

DEFINE_FWK_MODULE(ElectronHcalPFClusterIsolationProducer);
DEFINE_FWK_MODULE(PhotonHcalPFClusterIsolationProducer);
