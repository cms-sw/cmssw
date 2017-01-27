#ifndef EgammaIsolationRemappers_PhotonIDExternalProducerRemapper_h
#define EgammaIsolationRemappers_PhotonIDExternalProducerRemapper_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

class PhotonIDExternalProducerRemapper : public edm::stream::EDProducer<> {
 public:
  PhotonIDExternalProducerRemapper(const edm::ParameterSet&);
  ~PhotonIDExternalProducerRemapper();
 
  void produce(edm::Event&, const edm::EventSetup&);
 private:
  typedef typename edm::refhelper::ValueTrait<reco::PhotonCollection >::value Value;
  typedef edm::Ref<reco::PhotonCollection, Value, typename edm::refhelper::FindTrait<reco::PhotonCollection, Value>::value> CRef;
  typedef edm::ValueMap<CRef> CRefMap;

  edm::EDGetTokenT<reco::PhotonCollection > emObjectProducer_;
  edm::EDGetTokenT<CRefMap> newToOldObjectMap_;
  edm::EDGetTokenT<edm::ValueMap<bool> > idMap_;
};

#endif
