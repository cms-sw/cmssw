#ifndef RecoAlgos_ObjectSelector_h
#define RecoAlgos_ObjectSelector_h
/** \class ObjectSelector
 *
 * selects a subset of a collection. 
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.16 $
 *
 * $Id: ObjectSelector.h,v 1.16 2007/03/09 14:44:01 llista Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/UtilAlgos/interface/NonNullNumberSelector.h"
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>

namespace helper {

  template<typename Collection>
  struct IteratorToObjectConverter {
    typedef typename Collection::value_type value_type;
    template<typename I>
    static value_type convert( const I & i ) {
      return value_type( * * i );
    }
  };
  
  template<typename T>
   struct IteratorToObjectConverter<edm::OwnVector<T> > {
    typedef std::auto_ptr<T> value_type;
    template<typename I>
    static value_type convert( const I & i ) {
      return value_type( (*i)->clone() );
    }
  };

  template<typename C>
   struct IteratorToObjectConverter<edm::RefVector<C> > {
    typedef edm::Ref<C> value_type;
    template<typename I>
    static value_type convert( const I & i ) {
      return value_type( * i );
    }
  };

  template<typename OutputCollection, 
	   typename ClonePolicy = IteratorToObjectConverter<OutputCollection> >
  struct CollectionStoreManager {
    typedef OutputCollection collection;
    CollectionStoreManager() : selected_( new collection ) { }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & ) {
      for( I i = begin; i != end; ++ i ) {
	typename ClonePolicy::value_type v = ClonePolicy::convert( i );
        selected_->push_back( v );
      }
    }
    edm::OrphanHandle<collection> put( edm::Event & evt ) {
      return evt.put( selected_ );
    }
    size_t size() const { return selected_->size(); }
  private:
    std::auto_ptr<collection> selected_;
  };

  template<typename OutputCollection>
  struct NullPostProcessor {
    NullPostProcessor( const edm::ParameterSet & ) { }
    void init( edm::EDFilter & ) { }
    void process( edm::OrphanHandle<OutputCollection>, edm::Event & ) { }
  };

  template<typename OutputCollection>
  class ObjectSelectorBase : public edm::EDFilter {
    ObjectSelectorBase( const edm::ParameterSet & cfg ) {
      produces<OutputCollection>();
    }    
  };

  template<typename OutputCollection>
  struct StoreManagerTrait {
    typedef CollectionStoreManager<OutputCollection> type;
    typedef ObjectSelectorBase<OutputCollection> base;
  };

}

template<typename Selector, 
         typename OutputCollection = typename Selector::collection,
	 typename SizeSelector = NonNullNumberSelector,
	 typename PostProcessor = helper::NullPostProcessor<OutputCollection>,
	 typename StoreManager = typename helper::StoreManagerTrait<OutputCollection>::type,
	 typename Base = typename helper::StoreManagerTrait<OutputCollection>::base
	 >
class ObjectSelector : public Base {
public:
  /// constructor 
  explicit ObjectSelector( const edm::ParameterSet & cfg ) :
    Base( cfg ),
    src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
    filter_( false ),
    selector_( cfg ),
    sizeSelector_( reco::modules::make<SizeSelector>( cfg ) ),
    postProcessor_( cfg ) {
    const std::string filter( "filter" );
    std::vector<std::string> bools = cfg.template getParameterNamesForType<bool>();
    bool found = std::find( bools.begin(), bools.end(), filter ) != bools.end();
    if ( found ) filter_ = cfg.template getParameter<bool>( filter );
    postProcessor_.init( * this );
   }
  /// destructor
  virtual ~ObjectSelector() { }
  
private:
  /// process one event
  bool filter( edm::Event& evt, const edm::EventSetup& ) {
    edm::Handle<typename Selector::collection> source;
    evt.getByLabel( src_, source );
    StoreManager manager;
    selector_.select( source, evt );
    manager.cloneAndStore( selector_.begin(), selector_.end(), evt );
    bool result = ( ! filter_ || sizeSelector_( manager.size() ) );
    edm::OrphanHandle<OutputCollection> filtered = manager.put( evt );
    postProcessor_.process( filtered, evt );
    return result;
  }
  /// source collection label
  edm::InputTag src_;
  /// filter event
  bool filter_;
  /// Object collection selector
  Selector selector_;
  /// selected object collection size selector
  SizeSelector sizeSelector_;
  /// post processor
  PostProcessor postProcessor_;
};

#endif
