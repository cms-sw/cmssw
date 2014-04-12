#ifndef CandAlgos_CloneProducer_h
#define CandAlgos_CloneProducer_h
/** \class CloneProducer
 *
 * Clones a concrete Candidate collection
 * to a CandidateCollection (i.e.: OwnVector<Candidate>) filled
 * with clones of the original candidate collection
 *
 * \author: Francesco Fabozzi, INFN
 *          modified by Luca Lista, INFN
 *
 * Template parameters:
 * - C : Concrete candidate collection type
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"

template<typename C>
class CloneProducer : public edm::EDProducer {
public:
  /// constructor from parameter set
  explicit CloneProducer( const edm::ParameterSet& );
  /// destructor
  ~CloneProducer();

private:
  /// process an event
  virtual void produce( edm::Event&, const edm::EventSetup& );
  /// labels of the collection to be converted
  edm::EDGetTokenT<C> srcToken_;
};

template<typename C>
CloneProducer<C>::CloneProducer( const edm::ParameterSet& par ) :
  srcToken_( consumes<C>( par.template getParameter<edm::InputTag>( "src" ) ) ) {
  produces<reco::CandidateCollection>();
}

template<typename C>
CloneProducer<C>::~CloneProducer() {
}

template<typename C>
void CloneProducer<C>::produce( edm::Event& evt, const edm::EventSetup& ) {
  std::auto_ptr<reco::CandidateCollection> coll( new reco::CandidateCollection );
  edm::Handle<C> masterCollection;
  evt.getByToken( srcToken_, masterCollection );
  for( size_t i = 0; i < masterCollection->size(); ++i ) {
    reco::CandidateBaseRef masterClone( edm::Ref<C>( masterCollection, i ) );
    coll->push_back( masterClone->clone() );
  }
  evt.put( coll );
}

#endif
