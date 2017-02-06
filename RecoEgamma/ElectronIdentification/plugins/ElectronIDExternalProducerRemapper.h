#ifndef EgammaIsolationRemappers_ElectronIDExternalProducerRemapper_h
#define EgammaIsolationRemappers_ElectronIDExternalProducerRemapper_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class ElectronIDExternalProducerRemapper : public edm::stream::EDProducer<> {
 public:
  ElectronIDExternalProducerRemapper(const edm::ParameterSet&);
  ~ElectronIDExternalProducerRemapper();
 
  void produce(edm::Event&, const edm::EventSetup&);
 private:
  typedef typename edm::refhelper::ValueTrait<reco::GsfElectronCollection >::value Value;
  typedef edm::Ref<reco::GsfElectronCollection, Value, typename edm::refhelper::FindTrait<reco::GsfElectronCollection, Value>::value> CRef;
  typedef edm::ValueMap<CRef> CRefMap;

  edm::EDGetTokenT<reco::GsfElectronCollection > emObjectProducer_;
  edm::EDGetTokenT<CRefMap> newToOldObjectMap_;
  edm::EDGetTokenT<edm::ValueMap<float> > idMap_;
};

#endif
