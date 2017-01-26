#include "RecoEgamma/ElectronIdentification/plugins/PhotonIDExternalProducerRemapper.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

PhotonIDExternalProducerRemapper::PhotonIDExternalProducerRemapper(const edm::ParameterSet& config) : 
  emObjectProducer_       (consumes<reco::PhotonCollection >(config.getParameter<edm::InputTag>("candidateProducer"))),
  newToOldObjectMap_      (consumes<CRefMap>(config.getParameter<edm::InputTag>("newToOldObjectMap"))),
  idMap_           (consumes<edm::ValueMap<bool> >(config.getParameter<edm::InputTag>("idMap")))
{
  produces<edm::ValueMap<bool> >();
}

PhotonIDExternalProducerRemapper::~PhotonIDExternalProducerRemapper()
{
}

void PhotonIDExternalProducerRemapper::produce(edm::Event& iEvent, const edm::EventSetup&)
{
  edm::Handle<reco::PhotonCollection> emObjectHandle;
  iEvent.getByToken(emObjectProducer_, emObjectHandle);

  edm::Handle<CRefMap> objectMapHandle;
  iEvent.getByToken(newToOldObjectMap_, objectMapHandle);
  auto& objectMap(*objectMapHandle);

  edm::Handle<edm::ValueMap<bool> > idMapHandle;
  iEvent.getByToken(idMap_, idMapHandle);
  auto& idMap(*idMapHandle);

  std::vector<bool> retV(emObjectHandle->size(), 0.);

  for (unsigned iO(0); iO != emObjectHandle->size(); ++iO) {
    CRef ref(emObjectHandle, iO);
    auto oldRef(objectMap[ref]);
    retV[iO] = idMap[oldRef];
  }

  std::auto_ptr<edm::ValueMap<bool> > isoMap(new edm::ValueMap<bool>);
  edm::ValueMap<bool>::Filler filler(*isoMap);
  
  filler.insert(emObjectHandle, retV.begin(), retV.end());
  filler.fill();

  iEvent.put(isoMap);
}


DEFINE_FWK_MODULE(PhotonIDExternalProducerRemapper);
