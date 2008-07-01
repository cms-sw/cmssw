#include "DataFormats/HepMCCandidate/interface/FlavorHistory.h"

using namespace reco;
using namespace edm;
using namespace std;

FlavorHistory::FlavorHistory()
{
  flavorSource_ = FLAVOR_NULL;
}

FlavorHistory::FlavorHistory( FLAVOR_T flavorSource,
			      CandidatePtr const & parton,
			      CandidatePtr const & progenitor,
			      CandidatePtr const & sister ) :
  flavorSource_(flavorSource),
  parton_(parton),
  progenitor_(progenitor),
  sister_(sister)
{
  
}

FlavorHistory::FlavorHistory( FLAVOR_T flavorSource,
			      Handle<View<Candidate> > h_partons,
			      int parton,
			      int progenitor,
			      int sister ) :
  flavorSource_(flavorSource),
  parton_(h_partons,parton),
  progenitor_(h_partons,progenitor),
  sister_(h_partons,sister)
{
  
}

FlavorHistory::FlavorHistory( FLAVOR_T flavorSource,
			      Handle<CandidateCollection> h_partons,
			      int parton,
			      int progenitor,
			      int sister ) :
  flavorSource_(flavorSource),
  parton_(h_partons,parton),
  progenitor_(h_partons,progenitor),
  sister_(h_partons,sister)
{
  
}

