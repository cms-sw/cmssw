#ifndef CandAlgos_RecoToCandCollectionConverter_h
#define CandAlgos_RecoToCandCollectionConverter_h
/** \class RecoToCandCollectionConverter
 *
 * Clones a std::vector<> collection of reco candidates
 * to a OwnVector<Candidate> collection of candidates
 * 
 * Template parameters:
 * - R : RecoCandidate type
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include <string>

template<typename R>
class RecoToCandCollectionConverter : public edm::EDProducer {
public:
  /// constructor from parameter set
  explicit RecoToCandCollectionConverter( const edm::ParameterSet& );
  /// destructor
  ~RecoToCandCollectionConverter();

private:
  /// process an event
  virtual void produce( edm::Event&, const edm::EventSetup& );
  /// labels of the collection to be converted
  std::string src_;
};

template<typename R>
RecoToCandCollectionConverter<R>::RecoToCandCollectionConverter( const edm::ParameterSet& par ) : 
  src_( par.template getParameter<std::string>( "src" ) ) {
  produces<reco::CandidateCollection>();
}

template<typename R>
RecoToCandCollectionConverter<R>::~RecoToCandCollectionConverter() {
}

template<typename R>
void RecoToCandCollectionConverter<R>::produce( edm::Event& evt, const edm::EventSetup& ) {
  std::auto_ptr<reco::CandidateCollection> coll( new reco::CandidateCollection );
    edm::Handle<R> recoCollection;
    evt.getByLabel( src_, recoCollection );
    for( typename R::const_iterator r = recoCollection->begin(); r != recoCollection->end(); ++r ) {
      coll->push_back( (* r ).clone() );
    }
  evt.put( coll );
}

#endif
