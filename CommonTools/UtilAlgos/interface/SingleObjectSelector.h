#ifndef UtilAlgos_SingleObjectSelector_h
#define UtilAlgos_SingleObjectSelector_h
/* \class SingleObjectSelector
 *
 * \author Luca Lista, INFN
 */
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "CommonTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"



/* the following is just to ease transition
 *    grep -r SingleObjectSelector * | wc 
 *      209     540   22532
 */


template<typename InputCollection, typename Selector, 
	 typename EdmFilter,
	 typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<InputCollection>::type,
	 typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
	 typename PostProcessor = ::helper::NullPostProcessor<OutputCollection, EdmFilter>,
	 typename StoreManager = typename ::helper::StoreManagerTrait<OutputCollection, EdmFilter>::type,
	 typename Base = typename ::helper::StoreManagerTrait<OutputCollection, EdmFilter>::base,
	 typename RefAdder = typename ::helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
class SingleObjectSelectorBase : 
  public ObjectSelector<SingleElementCollectionSelector<InputCollection, Selector, OutputCollection, StoreContainer, RefAdder>, 
			OutputCollection, NonNullNumberSelector, PostProcessor, StoreManager, Base> {
public:
  // SingleObjectSelectorBase() = default;
  explicit SingleObjectSelectorBase( const edm::ParameterSet & cfg ) :
    ObjectSelector<SingleElementCollectionSelector<InputCollection, Selector, OutputCollection, StoreContainer, RefAdder>, 
		   OutputCollection, NonNullNumberSelector, PostProcessor, StoreManager, Base>( cfg ) { }
  virtual ~SingleObjectSelectorBase() { }
};



template<typename InputCollection, typename Selector,
	 typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<InputCollection>::type,
	 typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
	 typename PostProcessor = ::helper::NullPostProcessor<OutputCollection, edm::EDFilter> >
using SingleObjectSelectorLegacy = SingleObjectSelectorBase<InputCollection,Selector, edm::EDFilter, 
							    OutputCollection,StoreContainer,PostProcessor
							    >;



#include "FWCore/Framework/interface/stream/EDFilter.h"

template<typename InputCollection, typename Selector,
	 typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<InputCollection>::type,
	 typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
	 typename PostProcessor = ::helper::NullPostProcessor<OutputCollection, edm::stream::EDFilter<> > >
using SingleObjectSelectorStream = SingleObjectSelectorBase<InputCollection,Selector, edm::stream::EDFilter<>,
							    OutputCollection,StoreContainer,PostProcessor
							    >;


template<typename InputCollection, typename Selector,
	 typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<InputCollection>::type,
	 typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
	 typename PostProcessor = ::helper::NullPostProcessor<OutputCollection, edm::stream::EDFilter<> > >
using SingleObjectSelector = SingleObjectSelectorStream<InputCollection,Selector,
							OutputCollection,StoreContainer,PostProcessor>;



#endif

