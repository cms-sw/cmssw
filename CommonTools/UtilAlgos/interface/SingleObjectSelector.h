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

template<typename InputCollection, typename Selector, 
	 typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<InputCollection>::type,
	 typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
	 typename PostProcessor = ::helper::NullPostProcessor<OutputCollection>,
	 typename StoreManager = typename ::helper::StoreManagerTrait<OutputCollection>::type,
	 typename Base = typename ::helper::StoreManagerTrait<OutputCollection>::base,
	 typename RefAdder = typename ::helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
class SingleObjectSelector : 
  public ObjectSelector<SingleElementCollectionSelector<InputCollection, Selector, OutputCollection, StoreContainer, RefAdder>, 
			OutputCollection, NonNullNumberSelector, PostProcessor, StoreManager, Base> {
public:
  explicit SingleObjectSelector( const edm::ParameterSet & cfg ) :
    ObjectSelector<SingleElementCollectionSelector<InputCollection, Selector, OutputCollection, StoreContainer, RefAdder>, 
		   OutputCollection, NonNullNumberSelector, PostProcessor>( cfg ) { }
  virtual ~SingleObjectSelector() { }
};

#endif

