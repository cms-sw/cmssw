// $Id: CandSelectorBase.cc,v 1.3 2005/10/25 09:08:31 llista Exp $
#include <memory>
#include "PhysicsTools/CandAlgos/interface/CandSelectorBase.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Event.h"

using namespace aod;
using namespace edm;

CandSelectorBase::CandSelectorBase( const std::string & src, 
				    const boost::shared_ptr<CandSelector> & sel ) :
  select_( sel ), src_( src ) {
  produces<CandidateCollection>();
}

CandSelectorBase::~CandSelectorBase() {
}

void CandSelectorBase::produce( edm::Event& evt, const edm::EventSetup& ) {
 
  Handle<CandidateCollection> cands;
  evt.getByLabel( src_, cands );
  std::auto_ptr<CandidateCollection> comp( new CandidateCollection );
  for( CandidateCollection::const_iterator c = cands->begin(); c != cands->end(); ++c ) {
    std::auto_ptr<Candidate> cand( ( * c )->clone() );
    if( ( * select_ )( * cand ) ) {
      comp->push_back( cand.release() );
    }
  }
  evt.put( comp );
}

