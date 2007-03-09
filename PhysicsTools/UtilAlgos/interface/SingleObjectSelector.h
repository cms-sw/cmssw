#ifndef UtilAlgos_SingleObjectSelector_h
#define UtilAlgos_SingleObjectSelector_h
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

template<typename InputCollection, typename Selector, 
	 typename OutputCollection = InputCollection,
	 typename StoreContainer = std::vector<const typename InputCollection::value_type *>, 
	 typename PostProcessor = reco::helpers::NullPostProcessor<OutputCollection>,
	 typename RefAdder = typename helper::SelectionAdderTrait<StoreContainer>::type>
class SingleObjectSelector : 
  public ObjectSelector<SingleElementCollectionSelector<InputCollection, Selector, StoreContainer, RefAdder>, 
			OutputCollection, NonNullNumberSelector, PostProcessor,  
			helper::SimpleCollectionStoreManager<OutputCollection> > {
public:
  explicit SingleObjectSelector( const edm::ParameterSet & cfg ) :
    ObjectSelector<SingleElementCollectionSelector<InputCollection, Selector, StoreContainer, RefAdder>, 
		   OutputCollection, NonNullNumberSelector, PostProcessor,  
		   helper::SimpleCollectionStoreManager<OutputCollection> >( cfg ) { }
  virtual ~SingleObjectSelector() { }
};

#endif
