#ifndef CandAlgos_SingleObjectShallowCloneSelector_h
#define CandAlgos_SingleObjectShallowCloneSelector_h
/* \class SingleObjectShallowSelector
 *
 * \author Luca Lista, INFN
 */
#include "PhysicsTools/CandAlgos/interface/ObjectShallowCloneSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "PhysicsTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

template<typename InputCollection, typename Selector, 
	 typename StoreContainer = typename helper::StoreContainerTrait<reco::CandidateCollection>::type,
	 typename PostProcessor = helper::NullPostProcessor<reco::CandidateCollection>,
	 typename StoreManager = typename helper::StoreManagerTrait<reco::CandidateCollection>::type,
	 typename Base = typename helper::StoreManagerTrait<reco::CandidateCollection>::base,
	 typename RefAdder = typename helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
class SingleObjectShallowCloneSelector : 
  public ObjectShallowCloneSelector<SingleElementCollectionSelector<InputCollection, Selector, reco::CandidateCollection, 
                                                                    StoreContainer, RefAdder>, 
				    NonNullNumberSelector, PostProcessor> {
public:
  explicit SingleObjectShallowCloneSelector( const edm::ParameterSet & cfg ) :
    ObjectShallowCloneSelector<SingleElementCollectionSelector<InputCollection, Selector, reco::CandidateCollection, 
							       StoreContainer, RefAdder>, 
			       NonNullNumberSelector, PostProcessor>( cfg ) { }
  virtual ~SingleObjectShallowCloneSelector() { }
};

#endif
