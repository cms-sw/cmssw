#ifndef HepMCCandAlgos_MCTruthCompositeMatcher
#define HepMCCandAlgos_MCTruthCompositeMatcher
/* \class MCTruthCompositeMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCCandMatcher.h"

template<typename C>
class MCTruthCompositeMatcher : public edm::EDProducer {
public:
  explicit MCTruthCompositeMatcher( const edm::ParameterSet & );
  ~MCTruthCompositeMatcher();
private:
  typedef typename CandMatcher<C>::map_type map_type;
  edm::InputTag src_;
  std::vector<edm::InputTag> matchMaps_;
  void produce( edm::Event & , const edm::EventSetup & );
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

template<typename C>
MCTruthCompositeMatcher<C>::MCTruthCompositeMatcher( const edm::ParameterSet & cfg ) :
  src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
  matchMaps_( cfg.template getParameter<std::vector<edm::InputTag> >( "matchMaps" ) ) {
  produces<map_type>();
}

template<typename C>
MCTruthCompositeMatcher<C>::~MCTruthCompositeMatcher() {
}

template<typename C>
void MCTruthCompositeMatcher<C>::produce( edm::Event & evt , const edm::EventSetup & ) {
  using namespace edm;
  using namespace reco;
  using namespace std;
  typedef typename CandMatcher<C>::reference_type reference_type;
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
  MCCandMatcher<C> match( maps );
  auto_ptr<map_type> matchMap( new map_type );
  for( size_t i = 0; i != cands->size(); ++ i ) {
    const typename C::value_type & cand = ( * cands )[ i ];
    CandidateRef mc = match( cand );
    if ( mc.isNonnull() ) {
      matchMap->insert( reference_type( cands, i ), mc );      
    }
  }

  evt.put( matchMap );
}

#endif
