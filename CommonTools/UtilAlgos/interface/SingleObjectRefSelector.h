#ifndef UtilAlgos_SingleObjectRefSelector_h
#define UtilAlgos_SingleObjectRefSelector_h
/* \class SingleObjectRefSelector
 *
 * \author Luca Lista, INFN
 */
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "CommonTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionRefSelector.h"
#include "DataFormats/Common/interface/View.h"

template<typename InputType, typename Selector, 
	 typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<edm::View<InputType> >::type,
	 typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
	 typename PostProcessor = ::helper::NullPostProcessor<OutputCollection>,
	 typename StoreManager = typename ::helper::StoreManagerTrait<OutputCollection>::type,
	 typename Base = typename ::helper::StoreManagerTrait<OutputCollection>::base,
	 typename RefAdder = typename ::helper::SelectionAdderTrait<edm::View<InputType>, StoreContainer>::type>
class SingleObjectRefSelector : 
  public ObjectSelector<SingleElementCollectionRefSelector<InputType, Selector, OutputCollection, StoreContainer, RefAdder>, 
			OutputCollection, NonNullNumberSelector, PostProcessor, StoreManager, Base> {
public:
  explicit SingleObjectRefSelector(const edm::ParameterSet & cfg) :
    ObjectSelector<SingleElementCollectionRefSelector<InputType, Selector, OutputCollection, StoreContainer, RefAdder>, 
                   OutputCollection, NonNullNumberSelector, PostProcessor>( cfg ) { }
  virtual ~SingleObjectRefSelector() { }
};

#endif

