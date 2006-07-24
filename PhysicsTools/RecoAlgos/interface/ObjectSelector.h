#ifndef RecoAlgos_ObjectSelector_h
#define RecoAlgos_ObjectSelector_h
/** \class ObjectSelector
 *
 * selects a subset of a collection. 
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ObjectSelector.h,v 1.1 2006/07/24 10:09:08 llista Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/CloneTrait.h"
#include <utility>
#include <vector>
#include <memory>

namespace helper {
  template<typename C, typename P = typename edm::clonehelper::CloneTrait<C>::type>
  struct SimpleCollectionStoreManager {
    SimpleCollectionStoreManager() : selected_( new C ) { 
    }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & ) {
      for( I i = begin(); i != end; ++ i ) 
        selected_->push_back( P::clone( obj ) );
    }
    void put( edm::Event & evt ) {
      evt.put( selected_ );
    }
    bool empty() const { return selected_.empty(); }
  private:
    std::auto_ptr<C> selected_;
  };

  template<typename C>
    class ObjectSelectorBase : public edm::EDFilter {
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
  src_( cfg.template getParameter<std::string>( "src" ) ),
  filter_( cfg.template getParameter<bool>( "filter" ) ),
  selector_( cfg ) {
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
  std::string src_;
  /// filter event
  bool filter_;
  /// Object collection selector
  S selector_;
};



#endif
