#ifndef CommonTools_UtilAlgos_ObjectSelectorStream_h
#define CommonTools_UtilAlgos_ObjectSelectorStream_h
// -*- C++ -*-
//
// Package:     CommonTools/UtilAlgos
// Class  :     ObjectSelectorStream
// 
/**\class ObjectSelectorStream ObjectSelectorStream.h "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"

 Description: Template for constructing stream based object selector modules

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 30 May 2014 18:56:48 GMT
//

// system include files
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"

template<typename Selector,
         typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<typename Selector::collection>::type,
	 typename SizeSelector = NonNullNumberSelector,
  typename PostProcessor = ::helper::NullPostProcessor<OutputCollection, edm::stream::EDFilter<>>,
  typename StoreManager = typename ::helper::StoreManagerTrait<OutputCollection, edm::stream::EDFilter<>>::type,
  typename Init = typename ::reco::modules::EventSetupInit<Selector>::type
  >
  using ObjectSelectorStream = ObjectSelector<Selector,OutputCollection,SizeSelector, PostProcessor,StoreManager, typename ::helper::StoreManagerTrait<OutputCollection, edm::stream::EDFilter<>>::base, Init>;

#endif
