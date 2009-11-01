#ifndef UtilAlgos_SingleObjectRefSelector_h
#define UtilAlgos_SingleObjectRefSelector_h
/* \class SingleObjectRefSelector
 *
 * \author Luca Lista, INFN
 */
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "PhysicsTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionRefSelector.h"
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
