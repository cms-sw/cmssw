#ifndef RecoAlgos_ObjectSelector_h
#define RecoAlgos_ObjectSelector_h
/** \class ObjectSelector
 *
 * selects a subset of a collection.
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 * $Id: ObjectSelector.h,v 1.3 2010/02/20 20:55:27 wmtan Exp $
 *
 */

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorBase.h"
#include "CommonTools/UtilAlgos/interface/NonNullNumberSelector.h"
#include "CommonTools/UtilAlgos/interface/StoreManagerTrait.h"
#include "CommonTools/UtilAlgos/interface/SelectedOutputCollectionTrait.h"
#include "CommonTools/UtilAlgos/interface/NullPostProcessor.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"

template <typename Selector,
          typename OutputCollection =
              typename ::helper::SelectedOutputCollectionTrait<typename Selector::collection>::type,
          typename SizeSelector = NonNullNumberSelector,
          typename PostProcessor = ::helper::NullPostProcessor<OutputCollection>,
          typename StoreManager = typename ::helper::StoreManagerTrait<OutputCollection, edm::stream::EDFilter<>>::type,
          typename Base = typename ::helper::StoreManagerTrait<OutputCollection, edm::stream::EDFilter<>>::base,
          typename Init = typename ::reco::modules::EventSetupInit<Selector>::type>
using ObjectSelector =
    ObjectSelectorBase<Selector, OutputCollection, SizeSelector, PostProcessor, StoreManager, Base, Init>;

#endif
