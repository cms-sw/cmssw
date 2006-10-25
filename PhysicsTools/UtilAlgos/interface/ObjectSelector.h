#ifndef RecoAlgos_ObjectSelector_h
#define RecoAlgos_ObjectSelector_h
/** \class ObjectSelector
 *
 * selects a subset of a collection. 
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 * $Id: ObjectSelector.h,v 1.3 2006/07/26 09:10:47 llista Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/CloneTrait.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>

namespace helper {
  template<typename C, typename P = typename edm::clonehelper::CloneTrait<C>::type>
  struct SimpleCollectionStoreManager {
    SimpleCollectionStoreManager() : selected_( new C ) { 
    }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & ) {
      for( I i = begin; i != end; ++ i ) 
        selected_->push_back( P::clone( * * i ) );
    }
    void put( edm::Event & evt ) {
      evt.put( selected_ );
    }
    bool empty() const { return selected_->empty(); }
  private:
    std::auto_ptr<C> selected_;
  };

  template<typename C>
    struct ObjectSelectorBase : public edm::EDFilter {
      ObjectSelectorBase( const edm::ParameterSet & ) {
	produces<C>();
      }
   };

  template<typename C>
  struct CollectionStoreManager {
    typedef SimpleCollectionStoreManager<C> type;
    typedef ObjectSelectorBase<C> base;
  };
}

template<typename S, 
	 typename M = typename helper::CollectionStoreManager<typename S::collection>::type, 
	 typename B = typename helper::CollectionStoreManager<typename S::collection>::base>
class ObjectSelector : public B {
public:
  /// constructor 
  explicit ObjectSelector( const edm::ParameterSet & cfg ) :
  B( cfg ),
  src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
  filter_( false ),
  selector_( cfg ) {
    const std::string filter( "filter" );
    std::vector<std::string> bools = cfg.template getParameterNamesForType<bool>();
    bool found = std::find( bools.begin(), bools.end(), filter ) != bools.end();
    if ( found ) cfg.template getParameter<bool>( filter );
  }
  /// destructor
  virtual ~ObjectSelector() { }
  
private:
  /// process one event
  bool filter( edm::Event& evt, const edm::EventSetup& ) {
    edm::Handle<typename S::collection> source;
    evt.getByLabel( src_, source );
    M manager;
    selector_.select( * source, evt );
    manager.cloneAndStore( selector_.begin(), selector_.end(), evt );
    if ( filter_ && manager.empty() ) return false;
    manager.put( evt );
    return true;
  }
  /// source collection label
  edm::InputTag src_;
  /// filter event
  bool filter_;
  /// Object collection selector
  S selector_;
};



#endif
