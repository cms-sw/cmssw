#ifndef HepMCCandAlgos_MCTruthCompositeMatcher
#define HepMCCandAlgos_MCTruthCompositeMatcher
/* \class MCTruthCompositeMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCCandMatcher.h"

template<typename C1, typename C2 = C1>
class MCTruthCompositeMatcher : public edm::EDProducer {
public:
  explicit MCTruthCompositeMatcher( const edm::ParameterSet & );
  ~MCTruthCompositeMatcher();
private:
  typedef typename CandMatcher<C1, C2>::map_type map_type;
  edm::InputTag src_;
  std::vector<edm::InputTag> matchMaps_;
  void produce( edm::Event & , const edm::EventSetup&) override;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

template<typename C1, typename C2>
MCTruthCompositeMatcher<C1, C2>::MCTruthCompositeMatcher( const edm::ParameterSet & cfg ) :
  src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
  matchMaps_( cfg.template getParameter<std::vector<edm::InputTag> >( "matchMaps" ) ) {
  produces<map_type>();
}

template<typename C1, typename C2>
MCTruthCompositeMatcher<C1, C2>::~MCTruthCompositeMatcher() {
}

template<typename C1, typename C2>
void MCTruthCompositeMatcher<C1, C2>::produce( edm::Event & evt , const edm::EventSetup & ) {
  typedef typename CandMatcher<C1, C2>::reference_type reference_type;
  Handle<C1> cands;
  evt.getByLabel(src_, cands);
  
  size_t nMaps = matchMaps_.size();
  std::vector<const map_type *> maps;
  maps.reserve( nMaps );
  for( size_t i = 0; i != nMaps; ++ i ) {
    Handle<map_type> matchMap;
    evt.getByLabel( matchMaps_[i], matchMap );
    maps.push_back( & * matchMap );
  } 
  MCCandMatcher<C1, C2> match( maps );
  auto_ptr<map_type> matchMap( new map_type );
  for( size_t i = 0; i != cands->size(); ++ i ) {
    const typename C1::value_type & cand = ( * cands )[ i ];
    reference_type mc(match( cand ));
    if ( mc.isNonnull() ) {
      matchMap->insert( reference_type( cands, i ), mc );      
    }
  }

  evt.put( matchMap );
}

#endif

