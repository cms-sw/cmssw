#ifndef RecoAlgos_SingleElementCollectionSelectorPlusEvent_h
#define RecoAlgos_SingleElementCollectionSelectorPlusEvent_h
/** \class SingleElementCollectionSelectorPlusEvent
 *
 * selects a subset of a track collection based
 * on single element selection done via functor
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: SingleElementCollectionSelectorPlusEvent.h,v 1.1 2007/06/01 14:20:54 gpetrucc Exp $
 *
 */
#include "PhysicsTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "PhysicsTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

template<typename InputCollection, typename Selector, 
	 typename OutputCollection = typename helper::SelectedOutputCollectionTrait<InputCollection>::type, 
	 typename StoreContainer = typename helper::StoreContainerTrait<OutputCollection>::type,
	 typename RefAdder = typename helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
struct SingleElementCollectionSelectorPlusEvent {
  typedef InputCollection collection;
  typedef StoreContainer container;
  typedef typename container::const_iterator const_iterator;
  SingleElementCollectionSelectorPlusEvent( const edm::ParameterSet & cfg ) : 
    select_( reco::modules::make<Selector>( cfg ) ) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select(const edm::Handle<InputCollection> & c, const edm::Event &ev, const edm::EventSetup &) {
    selected_.clear();    
    for(size_t idx = 0; idx < c->size(); ++ idx) {
      if(select_((*c)[idx], ev)) 
	addRef_(selected_, c, idx);
    }
  }
private:
  StoreContainer selected_;
  Selector select_;
  RefAdder addRef_;
};

#endif
