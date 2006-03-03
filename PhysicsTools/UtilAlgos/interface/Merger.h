#ifndef UtilAlgos_Merger_h
#define UtilAlgos_Merger_h
/** \class Merger
 *
 * Merges an arbitrary number of collections 
 * into a single collection.
 * 
 * Template parameters:
 * - C : collection type
 * - P : policy class that specifies how objects 
 *       in the collection are are cloned
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: CandReducer.h,v 1.2 2006/03/03 10:20:44 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <vector>

template<typename C, typename P>
class Merger : public edm::EDProducer {
public:
  /// constructor from parameter set
  explicit Merger( const edm::ParameterSet& );
  /// destructor
  ~Merger();

private:
  /// process an event
  virtual void produce( edm::Event&, const edm::EventSetup& );
  /// vector of strings
  typedef std::vector<std::string> vstring;
  /// labels of the collections to be merged
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
