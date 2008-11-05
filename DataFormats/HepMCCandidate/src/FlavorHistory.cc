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
			      CandidatePtr const & sister,
			      reco::ShallowClonePtrCandidate const & matchedJet,
			      reco::ShallowClonePtrCandidate const & sisterJet ) :
  flavorSource_(flavorSource),
  parton_(parton),
  progenitor_(progenitor),
  sister_(sister),
  matchedJet_(matchedJet),
  sisterJet_(sisterJet)
{
  
}

FlavorHistory::FlavorHistory( FLAVOR_T flavorSource,
			      Handle<View<Candidate> > h_partons,
			      int parton,
			      int progenitor,
			      int sister,
			      reco::ShallowClonePtrCandidate const & matchedJet,
			      reco::ShallowClonePtrCandidate const & sisterJet ) :
  flavorSource_(flavorSource),
  parton_    ( parton     >= 0 && static_cast<unsigned int>(parton)     < h_partons->size() ? CandidatePtr(h_partons,parton)     : CandidatePtr()),
  progenitor_( progenitor >= 0 && static_cast<unsigned int>(progenitor) < h_partons->size() ? CandidatePtr(h_partons,progenitor) : CandidatePtr()),
  sister_    ( sister     >= 0 && static_cast<unsigned int>(sister)     < h_partons->size() ? CandidatePtr(h_partons,sister)     : CandidatePtr()),
  matchedJet_( matchedJet ),
  sisterJet_ ( sisterJet )
{
  
}

FlavorHistory::FlavorHistory( FLAVOR_T flavorSource,
			      Handle<CandidateCollection> h_partons,
			      int parton,
			      int progenitor,
			      int sister,
			      reco::ShallowClonePtrCandidate const & matchedJet,
			      reco::ShallowClonePtrCandidate const & sisterJet ) :
  flavorSource_(flavorSource),
  parton_    ( parton     >= 0 && static_cast<unsigned int>(parton)     < h_partons->size() ? CandidatePtr(h_partons,parton)     : CandidatePtr()),
  progenitor_( progenitor >= 0 && static_cast<unsigned int>(progenitor) < h_partons->size() ? CandidatePtr(h_partons,progenitor) : CandidatePtr()),
  sister_    ( sister     >= 0 && static_cast<unsigned int>(sister)     < h_partons->size() ? CandidatePtr(h_partons,sister)     : CandidatePtr()),
  matchedJet_( matchedJet ),
  sisterJet_ ( sisterJet )
{
  
}

