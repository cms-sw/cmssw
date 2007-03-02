#ifndef UtilAlgos_SingleObjectSelector_h
#define UtilAlgos_SingleObjectSelector_h
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

template<typename C, typename S, 
	 typename SC = std::vector<const typename C::value_type *>, 
	 typename A = typename helper::SelectionAdderTrait<SC>::type>
class SingleObjectSelector : 
  public ObjectSelector<SingleElementCollectionSelector<C, S, SC, A> > {
public:
  explicit SingleObjectSelector( const edm::ParameterSet & cfg ) :
    ObjectSelector<SingleElementCollectionSelector<C, S, SC, A> >( cfg ) { }
  virtual ~SingleObjectSelector() { }
};

#endif
