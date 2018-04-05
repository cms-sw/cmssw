#ifndef CommonTools_UtilAlgos_ObjectSelectorStreamProducer_h
#define CommonTools_UtilAlgos_ObjectSelectorStreamProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorProducer.h"

template<typename Selector,
         typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<typename Selector::collection>::type,
         typename PostProcessor = ::helper::NullPostProcessor<OutputCollection, edm::stream::EDProducer<>>,
         typename StoreManager = typename ::helper::StoreManagerTrait<OutputCollection, edm::stream::EDProducer<>>::type,
         typename Init = typename ::reco::modules::EventSetupInit<Selector>::type
         >
  using ObjectSelectorStreamProducer = ObjectSelectorProducer<Selector, OutputCollection, PostProcessor, StoreManager, typename ::helper::StoreManagerTrait<OutputCollection, edm::stream::EDProducer<>>::base, Init>;

#endif
