#ifndef RecoAlgos_SingleElementCollectionSelector_h
#define RecoAlgos_SingleElementCollectionSelector_h
/** \class SingleElementCollectionSelector
 *
 * selects a subset of a track collection based
 * on single element selection done via functor
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.5 $
 *
 * $Id: SingleElementCollectionSelector.h,v 1.5 2007/01/31 14:51:37 llista Exp $
 *
 */
#include "PhysicsTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

template<typename InputCollection, typename Selector, 
	 typename StoreContainer = std::vector<const typename InputCollection::value_type *>, 
	 typename RefAdder = typename helper::SelectionAdderTrait<StoreContainer>::type>
struct SingleElementCollectionSelector {
  typedef InputCollection collection;
  typedef StoreContainer container;
  typedef typename container::const_iterator const_iterator;
  SingleElementCollectionSelector( const edm::ParameterSet & cfg ) : 
    select_( reco::modules::make<Selector>( cfg ) ) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select( const edm::Handle<InputCollection> & c, const edm::Event & ) {
    selected_.clear();    
    for( size_t idx = 0; idx < c->size(); ++ idx ) {
      if ( select_( ( * c )[ idx ] ) ) 
	addRef_( selected_, c, idx );
    }
  }
private:
  StoreContainer selected_;
  Selector select_;
  RefAdder addRef_;
};

#endif
