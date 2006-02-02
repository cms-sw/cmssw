#ifndef UtilAlgos_SelectorProducer_h
#define UtilAlgos_SelectorProducer_h
// Candidate Selector base producer module
// $Id: SelectorProducer.h,v 1.3 2005/10/25 09:08:31 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "boost/shared_ptr.hpp"
#include <string>
#include <memory>

template<typename C, typename S, typename P>
class SelectorProducer : public edm::EDProducer {
public:
  explicit SelectorProducer( const std::string &,
			     const boost::shared_ptr<S> & =
			     boost::shared_ptr<S>() );
  ~SelectorProducer();
  
protected:
  boost::shared_ptr<S> select_;

private:
  virtual void produce( edm::Event&, const edm::EventSetup& );
  std::string src_;
};

template<typename C, typename S, typename P>
SelectorProducer<C, S, P>::SelectorProducer( const std::string & src, 
					     const boost::shared_ptr<S> & sel ) :
  select_( sel ), src_( src ) {
  produces<C>();
}

template<typename C, typename S, typename P>
SelectorProducer<C, S, P>::~SelectorProducer() {
}

template<typename C, typename S, typename P>
void SelectorProducer<C, S, P>::produce( edm::Event& evt, const edm::EventSetup& ) {
  edm::Handle<C> coll;
  evt.getByLabel( src_, coll );
  std::auto_ptr<C> sel( new C );
  for( typename C::const_iterator c = coll->begin(); c != coll->end(); ++c ) {
    std::auto_ptr<typename C::value_type> o( P::clone( * c ) );
    if( ( * select_ )( * o ) ) {
      sel->push_back( o.release() );
    }
  }
  evt.put( sel );
}


#endif
