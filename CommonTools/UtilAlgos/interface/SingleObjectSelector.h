#ifndef UtilAlgos_SingleObjectSelector_h
#define UtilAlgos_SingleObjectSelector_h
/* \class SingleObjectSelector
 *
 * \author Luca Lista, INFN
 */
#include "CommonTools/UtilAlgos/interface/ObjectSelectorBase.h"
#include "CommonTools/UtilAlgos/interface/NonNullNumberSelector.h"
#include "CommonTools/UtilAlgos/interface/StoreManagerTrait.h"
#include "CommonTools/UtilAlgos/interface/SelectedOutputCollectionTrait.h"
#include "CommonTools/UtilAlgos/interface/NullPostProcessor.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include "CommonTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "CommonTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

/* the following is just to ease transition
 *    grep -r SingleObjectSelector * | wc 
 *      209     540   22532
 */

template <typename InputCollection,
          typename Selector,
          typename EdmFilter,
          typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<InputCollection>::type,
          typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
          typename PostProcessor = ::helper::NullPostProcessor<OutputCollection>,
          typename StoreManager = typename ::helper::StoreManagerTrait<OutputCollection, EdmFilter>::type,
          typename Base = typename ::helper::StoreManagerTrait<OutputCollection, EdmFilter>::base,
          typename RefAdder = typename ::helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
class SingleObjectSelectorBase
    : public ObjectSelectorBase<
          SingleElementCollectionSelector<InputCollection, Selector, OutputCollection, StoreContainer, RefAdder>,
          OutputCollection,
          NonNullNumberSelector,
          PostProcessor,
          StoreManager,
          Base,
          typename ::reco::modules::EventSetupInit<
              SingleElementCollectionSelector<InputCollection, Selector, OutputCollection, StoreContainer, RefAdder>>::
              type> {
  using Init = typename ::reco::modules::EventSetupInit<
      SingleElementCollectionSelector<InputCollection, Selector, OutputCollection, StoreContainer, RefAdder>>::type;

public:
  // SingleObjectSelectorBase() = default;
  explicit SingleObjectSelectorBase(const edm::ParameterSet& cfg)
      : ObjectSelectorBase<
            SingleElementCollectionSelector<InputCollection, Selector, OutputCollection, StoreContainer, RefAdder>,
            OutputCollection,
            NonNullNumberSelector,
            PostProcessor,
            StoreManager,
            Base,
            Init>(cfg) {}
  ~SingleObjectSelectorBase() override {}
};

#include "FWCore/Framework/interface/stream/EDFilter.h"

template <typename InputCollection,
          typename Selector,
          typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<InputCollection>::type,
          typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
          typename PostProcessor = ::helper::NullPostProcessor<OutputCollection>>
using SingleObjectSelectorStream = SingleObjectSelectorBase<InputCollection,
                                                            Selector,
                                                            edm::stream::EDFilter<>,
                                                            OutputCollection,
                                                            StoreContainer,
                                                            PostProcessor>;

template <typename InputCollection,
          typename Selector,
          typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<InputCollection>::type,
          typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
          typename PostProcessor = ::helper::NullPostProcessor<OutputCollection>>
using SingleObjectSelector =
    SingleObjectSelectorStream<InputCollection, Selector, OutputCollection, StoreContainer, PostProcessor>;

#endif
