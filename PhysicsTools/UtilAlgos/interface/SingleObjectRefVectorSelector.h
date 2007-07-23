#ifndef UtilAlgos_SingleObjectRefVectorSelector_h
#define UtilAlgos_SingleObjectRefVectorSelector_h
#include "PhysicsTools/UtilAlgos/interface/ObjectRefVectorSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

template<typename InputCollection, typename Selector>
class SingleObjectRefVectorSelector : 
  public ObjectRefVectorSelector<SingleElementCollectionSelector<InputCollection, Selector, edm::RefVector<InputCollection> > > {
public:
  explicit SingleObjectRefVectorSelector( const edm::ParameterSet & cfg ) :
    ObjectRefVectorSelector<SingleElementCollectionSelector<InputCollection, Selector, edm::RefVector<InputCollection> > >( cfg ) { }
  virtual ~SingleObjectRefVectorSelector() { }
};

#endif
