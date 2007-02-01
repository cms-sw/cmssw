/* \class MCTruthCompositeMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class MCTruthCompositeMatcher : public edm::EDProducer {
public:
  explicit MCTruthCompositeMatcher( const edm::ParameterSet & );
  ~MCTruthCompositeMatcher();
private:
  edm::InputTag src_;
  edm::InputTag matchMap_;
  void produce( edm::Event & , const edm::EventSetup & );
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCCandMatcher.h"
using namespace edm;
using namespace reco;
using namespace std;

MCTruthCompositeMatcher::MCTruthCompositeMatcher( const ParameterSet & cfg ) :
  src_( cfg.getParameter<InputTag>( "src" ) ),
  matchMap_( cfg.getParameter<InputTag>( "matchMap" ) ) {
  produces<CandMatchMap>();
}

MCTruthCompositeMatcher::~MCTruthCompositeMatcher() {
}

void MCTruthCompositeMatcher::produce( edm::Event & evt , const edm::EventSetup & ) {
  Handle<CandidateCollection> cands;  
  evt.getByLabel( src_, cands ) ;
  Handle<CandMatchMap> mcMatchMap;
  evt.getByLabel( matchMap_, mcMatchMap );

  MCCandMatcher match( * mcMatchMap );
  auto_ptr<CandMatchMap> matchMap( new CandMatchMap );

  for( size_t i = 0; i != cands->size(); ++ i ) {
    const Candidate & cand = ( * cands )[ i ];
    CandidateRef mc = match( cand );
    if ( mc.isNonnull() ) {
      matchMap->insert( CandidateRef( cands, i ), mc );      
    }
  }

  evt.put( matchMap );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MCTruthCompositeMatcher );
