#ifndef UtilAlgos_SingleObjectSelector_h
#define UtilAlgos_SingleObjectSelector_h
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"

namespace helper {
  template<typename OutputCollection>
    struct StoreContainerTrait {
      typedef std::vector<const typename OutputCollection::value_type *> type;
  };

  template<typename C>
  struct StoreContainerTrait<edm::RefVector<C> > {
    typedef edm::RefVector<C> type;
  };

  template<typename R, typename C>
   struct StoreContainerTrait<edm::AssociationVector<R, C> > {
     typedef typename StoreContainerTrait<typename R::product_type>::type type;
  };
}

template<typename InputCollection, typename Selector, 
	 typename OutputCollection = InputCollection,
	 typename StoreContainer = typename helper::StoreContainerTrait<OutputCollection>::type,
	 typename PostProcessor = helper::NullPostProcessor<OutputCollection>,
	 typename StoreManager = typename helper::StoreManagerTrait<OutputCollection>::type,
	 typename Base = typename helper::StoreManagerTrait<OutputCollection>::base,
	 typename RefAdder = typename helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
class SingleObjectSelector : 
  public ObjectSelector<SingleElementCollectionSelector<InputCollection, Selector, StoreContainer, RefAdder>, 
			OutputCollection, NonNullNumberSelector, PostProcessor, StoreManager, Base> {
public:
  explicit SingleObjectSelector( const edm::ParameterSet & cfg ) :
    ObjectSelector<SingleElementCollectionSelector<InputCollection, Selector, StoreContainer, RefAdder>, 
		   OutputCollection, NonNullNumberSelector, PostProcessor>( cfg ) { }
  virtual ~SingleObjectSelector() { }
};

#endif
