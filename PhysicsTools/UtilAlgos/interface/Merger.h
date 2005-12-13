#ifndef CandAlgos_Merger_h
#define CandAlgos_Merger_h
// Merges multiple collections
// $Id: Merger.h,v 1.1 2005/12/13 01:47:45 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <vector>

template<typename C, typename P>
class Merger : public edm::EDProducer {
public:
  explicit Merger( const edm::ParameterSet& );
  ~Merger();

private:
  virtual void produce( edm::Event&, const edm::EventSetup& );
  typedef std::vector<std::string> vstring;
  vstring src_;
};

template<typename C, typename P>
Merger<C, P>::Merger( const edm::ParameterSet& par ) : 
  src_( par.template getParameter<vstring>( "src" ) ) {
  produces<C>();
}

template<typename C, typename P>
Merger<C, P>::~Merger() {
}

template<typename C, typename P>
void Merger<C, P>::produce( edm::Event& evt, const edm::EventSetup& ) {
  std::auto_ptr<C> coll( new C );
  for( vstring::const_iterator s = src_.begin(); s != src_.end(); ++ s ) {
    edm::Handle<C> h;
    evt.getByLabel( * s, h );
    for( typename C::const_iterator c = h->begin(); c != h->end(); ++c ) {
      coll->push_back( P::clone( * c ) );
    }
  }
  evt.put( coll );
}

#endif
