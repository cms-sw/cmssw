#ifndef UtilAlgos_CollectionRecoverer_h
#define UtilAlgos_CollectionRecoverer_h
/** \class CollectionRecoverer
 *
 * Creates an empty collection for events 
 * where a collection is missing. Leaves the
 * existing collection if already present in the event.
 * 
 * Template parameters:
 * - C : collection type
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.8 $
 *
 * $Id: ObjectCounter.h,v 1.8 2006/04/27 13:11:26 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

template<typename C, typename P>
class CollectionRecoverer : public edm::EDProducer {
public:
  /// constructor from parameter set
  explicit CollectionRecoverer( const edm::ParameterSet& );
  /// end-of-job processing
  void endJob();
  
private:
  /// event processing
  virtual void produce( edm::Event&, const edm::EventSetup& );
  /// label of source collection
  std::string src_;
  /// count how many missing collections
  unsigned long nMissing_;
  /// count how many processed events
  unsigned long n_;
};

template<typename C, typename P>
CollectionRecoverer<C, P>::CollectionRecoverer( const edm::ParameterSet& par ) : 
  src_( par.template getParameter<std::string>( "src" ) ) {
}

template<typename C, typename P>
void CollectionRecoverer<C, P>::endJob() {
  if ( nMissing_ > 0 )
    edm::LogWarning ( "CollectionRecoverer" ) << "Recovered " << nMissing_ << "/" << n_ << " collections";
}

template<typename C, typename P>
void CollectionRecoverer<C, P>::produce( edm::Event& evt, const edm::EventSetup& ) {
  std::auto_ptr<C> coll( new C );
  edm::Handle<C> h;
  try {
    evt.getByLabel( src_, h );
    for( typename C::const_iterator c = h->begin(); c != h->end(); ++c ) {
      coll->push_back( P::clone( * c ) );
    }    
  } catch ( ... ) {
    ++ nMissing_;
  }
  evt.put( coll );
  ++ n_;
 }

#endif
