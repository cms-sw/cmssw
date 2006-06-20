#ifndef UtilAlgos_SelectorProducer_h
#define UtilAlgos_SelectorProducer_h
/** \class SelectorProducer
 *
 * selects objects from a collection according to a criterion 
 * specified by a functor class and saves their clones in a new 
 * collection.
 * 
 * Template parameters:
 * - C : collection type
 * - S : selector function
 * - P : policy class that specifies how objects 
 *       in the collection are are cloned
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.4 $
 *
 * $Id: SelectorProducer.h,v 1.4 2006/06/14 11:54:32 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/CloneTrait.h"
#include "boost/shared_ptr.hpp"
#include <string>
#include <memory>

template<typename C, typename S, typename P = typename edm::clonehelper::CloneTrait<C>::type >
class SelectorProducerBase : public edm::EDProducer {
public:
  /// constructor 
  explicit SelectorProducerBase( const edm::ParameterSet &, const boost::shared_ptr<S> & = boost::shared_ptr<S>() );
  /// destructor
  virtual ~SelectorProducerBase();
  
protected:
  /// selector object
  boost::shared_ptr<S> select_;

private:
  /// process one event
  virtual void produce( edm::Event&, const edm::EventSetup& );
  /// source collection label
  std::string src_;
};

template<typename C, typename S, typename P>
SelectorProducerBase<C, S, P>::SelectorProducerBase( const edm::ParameterSet & cfg, 
						     const boost::shared_ptr<S> & sel ) :
  select_( sel ), src_( cfg.template getParameter<std::string>( "src" ) ) {
  produces<C>();
}

template<typename C, typename S, typename P>
SelectorProducerBase<C, S, P>::~SelectorProducerBase() {
}

template<typename C, typename S, typename P>
void SelectorProducerBase<C, S, P>::produce( edm::Event& evt, const edm::EventSetup& ) {
  edm::Handle<C> coll;
  evt.getByLabel( src_, coll );
  std::auto_ptr<C> sel( new C );
  for( typename C::const_iterator c = coll->begin(); c != coll->end(); ++ c ) {
    if( ( * select_ )( * c ) ) 
      sel->push_back( P::clone( * c ) );
  }
  evt.put( sel );
}

template<typename C, typename S, typename P = typename edm::clonehelper::CloneTrait<C>::type >
class SelectorProducer : public SelectorProducerBase<C, S, P> {
public:
  explicit SelectorProducer( const edm::ParameterSet & cfg ) :
    SelectorProducerBase<C, S, P>( cfg, boost::shared_ptr<S>( new S ) ) { }
};


#endif
