// $Id: CandSelector.cc,v 1.3 2005/10/21 15:10:13 llista Exp $
#include <memory>
#include "PhysicsTools/CandAlgos/interface/CandSelectorBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/Selector.h"

using namespace aod;
using namespace edm;
typedef Candidate::collection Candidates;

CandSelectorBase::CandSelectorBase( const std::string & src, 
				    const boost::shared_ptr<aod::Selector> & sel ) :
  select_( sel ), src_( src ) {
  produces<Candidates>();
}

CandSelectorBase::~CandSelectorBase() {
}

void CandSelectorBase::produce( edm::Event& evt, const edm::EventSetup& ) {
 
  Handle<Candidates> cands;
  evt.getByLabel( src_, cands );
  std::auto_ptr<Candidates> comp( new Candidates );
  for( Candidates::const_iterator c = cands->begin(); c != cands->end(); ++c ) {
    std::auto_ptr<Candidate> cand( ( * c )->clone() );
    if( ( * select_ )( * cand ) ) {
      comp->push_back( cand.release() );
    }
  }
  evt.put( comp );
}

