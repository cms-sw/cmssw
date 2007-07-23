#ifndef RecoAlgos_SingleElementCollectionSelector_h
#define RecoAlgos_SingleElementCollectionSelector_h
/** \class SingleElementCollectionSelector
 *
 * selects a subset of a collection based
 * on single element selection done via functor
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.7 $
 *
 * $Id: SingleElementCollectionSelector.h,v 1.7 2007/05/15 16:07:52 llista Exp $
 *
 */
#include "PhysicsTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "PhysicsTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

template<typename InputCollection, typename Selector, 
	 typename OutputCollection = typename helper::SelectedOutputCollectionTrait<InputCollection>::type, 
	 typename StoreContainer = typename helper::StoreContainerTrait<OutputCollection>::type,
	 typename RefAdder = typename helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
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
