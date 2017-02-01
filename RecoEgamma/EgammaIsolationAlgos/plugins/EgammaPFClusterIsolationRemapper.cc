#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaPFClusterIsolationRemapper.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

template<typename C>
EgammaPFClusterIsolationRemapper<C>::EgammaPFClusterIsolationRemapper(const edm::ParameterSet& config) : 
  emObjectProducer_       (consumes<C>(config.getParameter<edm::InputTag>("candidateProducer"))),
  newToOldObjectMap_      (consumes<CRefMap>(config.getParameter<edm::InputTag>("newToOldObjectMap"))),
  isolationMap_           (consumes<FloatMap>(config.getParameter<edm::InputTag>("isolationMap")))
{
  produces<edm::ValueMap<float>>();
}

template<typename C>
EgammaPFClusterIsolationRemapper<C>::~EgammaPFClusterIsolationRemapper()
{
}

template<typename C>
void EgammaPFClusterIsolationRemapper<C>::produce(edm::Event& iEvent, const edm::EventSetup&)
{
  edm::Handle<C> emObjectHandle;
  iEvent.getByToken(emObjectProducer_, emObjectHandle);

  edm::Handle<CRefMap> objectMapHandle;
  iEvent.getByToken(newToOldObjectMap_, objectMapHandle);
  auto& objectMap(*objectMapHandle);

  edm::Handle<FloatMap> isolationMapHandle;
  iEvent.getByToken(isolationMap_, isolationMapHandle);
  auto& isolationMap(*isolationMapHandle);

  std::vector<float> retV(emObjectHandle->size(), 0.);

  for (unsigned iO(0); iO != emObjectHandle->size(); ++iO) {
    CRef ref(emObjectHandle, iO);
    auto oldRef(objectMap[ref]);
    retV[iO] = isolationMap[oldRef];
  }

  std::auto_ptr<FloatMap> isoMap(new FloatMap());
  FloatMap::Filler filler(*isoMap);
  
  filler.insert(emObjectHandle, retV.begin(), retV.end());
  filler.fill();

  iEvent.put(isoMap);
}

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

typedef EgammaPFClusterIsolationRemapper<reco::GsfElectronCollection> ElectronPFClusterIsolationRemapper;
typedef EgammaPFClusterIsolationRemapper<reco::PhotonCollection> PhotonPFClusterIsolationRemapper;

DEFINE_FWK_MODULE(ElectronPFClusterIsolationRemapper);
DEFINE_FWK_MODULE(PhotonPFClusterIsolationRemapper);
