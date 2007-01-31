#ifndef RecoAlgos_SingleElementCollectionSelector_h
#define RecoAlgos_SingleElementCollectionSelector_h
/** \class SingleElementCollectionSelector
 *
 * selects a subset of a track collection based
 * on single element selection done via functor
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.4 $
 *
 * $Id: SingleElementCollectionSelector.h,v 1.4 2006/10/27 10:03:45 llista Exp $
 *
 */
#include "PhysicsTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

template<typename C, typename S, 
	 typename SC = std::vector<const typename C::value_type *>, 
	 typename A = typename helper::SelectionAdderTrait<SC>::type>
struct SingleElementCollectionSelector {
  typedef C collection;
  typedef SC container;
  typedef typename container::const_iterator const_iterator;
  SingleElementCollectionSelector( const edm::ParameterSet & cfg ) : 
    select_( reco::modules::make<S>( cfg ) ) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select( const edm::Handle<C> & c, const edm::Event & ) {
    selected_.clear();    
    for( size_t idx = 0; idx < c->size(); ++ idx ) {
      if ( select_( ( * c )[ idx ] ) ) 
	A::add( selected_, c, idx );
    }
  }
private:
  container selected_;
  S select_;
};

#endif
