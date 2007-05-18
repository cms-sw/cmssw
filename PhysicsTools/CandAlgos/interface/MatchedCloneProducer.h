#ifndef HepMCCandAlgos_MatchedCloneProducer
#define HepMCCandAlgos_MatchedCloneProducer
/* \class MatchedCloneProducer
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "PhysicsTools/CandUtils/interface/FastCandMatcher.h"

template<typename C>
class MatchedCloneProducer : public edm::EDProducer {
public:
  explicit MatchedCloneProducer( const edm::ParameterSet & );
  ~MatchedCloneProducer();
private:
  typedef typename FastCandMatcher<C>::map_type map_type;
  edm::InputTag src_;
  std::vector<edm::InputTag> matchMaps_;
  void produce( edm::Event & , const edm::EventSetup & );
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

template<typename C>
MatchedCloneProducer<C>::MatchedCloneProducer( const edm::ParameterSet & cfg ) :
  src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
  matchMaps_( cfg.template getParameter<std::vector<edm::InputTag> >( "matchMaps" ) ) {
  produces<reco::CandidateCollection>();
}

template<typename C>
MatchedCloneProducer<C>::~MatchedCloneProducer() {
}

template<typename C>
void MatchedCloneProducer<C>::produce( edm::Event & evt , const edm::EventSetup & ) {
  using namespace edm;
  using namespace reco;
  using namespace std;
  Handle<C> cands;
  evt.getByLabel( src_, cands );
  
  size_t nMaps = matchMaps_.size();
  std::vector<const map_type *> maps;
  maps.reserve( nMaps );
  for( size_t i = 0; i != nMaps; ++ i ) {
    Handle<map_type> matchMap;
    evt.getByLabel( matchMaps_[i], matchMap );
    maps.push_back( & * matchMap );
  } 
  FastCandMatcher<C> match( maps );
  auto_ptr<CandidateCollection> matchMap( new CandidateCollection );
  for( CandidateCollection::const_iterator i = cands->begin(); i != cands->end(); ++ i ) {
    const Candidate * matched = match( * i );
    if ( matched != 0 )
      matchMap->push_back( matched->clone() );      
  }

  evt.put( matchMap );
}

#endif
