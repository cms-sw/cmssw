#include "DataFormats/HepMCCandidate/interface/FlavorHistoryEvent.h"
#include "TLorentzVector.h"

#include <map>

using namespace reco;
using namespace std;

// Loop over flavor histories, count number of genjets with
// flavor b, c, or l
void FlavorHistoryEvent::cache()
{
  // Set cached to false
  cached_ = false;
  nb_ = nc_ = 0;
  highestFlavor_ = 0;
  // get iterators to the history vector
  const_iterator i = histories_.begin(),
    ibegin = histories_.begin(),
    iend = histories_.end();
  // loop over the history vector and count the number of 
  // partons of flavors "b" and "c" that have a matched genjet.
  for ( ; i != iend; ++i ) {
    FlavorHistory const & flavHist = *i;
    CandidatePtr const & parton = flavHist.parton();
    int pdgId = -1;
    if ( parton.isNonnull() ) pdgId = abs(parton->pdgId());
    ShallowClonePtrCandidate const & matchedJet = flavHist.matchedJet();
    if ( matchedJet.masterClonePtr().isNonnull() ) {
      if ( pdgId == 5 ) nb_++;
      if ( pdgId == 4 ) nc_++;
    }
  }

  std::pair<double,int> p = calculateDR();
  dR_ = p.first;
  highestFlavor_ = p.second;

  // now we're cached, can return values quickly
  cached_ = true;
}



// NOTE: This is present such that if the user has
// an uncached const version, you can still access
// the heavy flavor content.
//
// Loop over flavor histories, count number of genjets with
// flavor b. 
unsigned int FlavorHistoryEvent::calculateNB() const
{

  unsigned int nb= 0;
  // get iterators to the history vector
  const_iterator i = histories_.begin(),
    ibegin = histories_.begin(),
    iend = histories_.end();
  // loop over the history vector and count the number of 
  // partons of flavors "b" and "c" that have a matched genjet.
  for ( ; i != iend; ++i ) {
    FlavorHistory const & flavHist = *i;
    CandidatePtr const & parton = flavHist.parton();
    int pdgId = -1;
    if ( parton.isNonnull() ) pdgId = abs(parton->pdgId());
    ShallowClonePtrCandidate const & matchedJet = flavHist.matchedJet();
    if ( matchedJet.masterClonePtr().isNonnull() ) {
      if ( pdgId == 5 ) nb++;
    }
  }
  return nb;
}


// NOTE: This is present such that if the user has
// an uncached const version, you can still access
// the heavy flavor content.
//
// Loop over flavor histories, count number of genjets with
// flavor c. 
unsigned int FlavorHistoryEvent::calculateNC() const
{

  unsigned int nc= 0;
  // get iterators to the history vector
  const_iterator i = histories_.begin(),
    ibegin = histories_.begin(),
    iend = histories_.end();
  // loop over the history vector and count the number of 
  // partons of flavors "b" and "c" that have a matched genjet.
  for ( ; i != iend; ++i ) {
    FlavorHistory const & flavHist = *i;
    CandidatePtr const & parton = flavHist.parton();
    int pdgId = -1;
    if ( parton.isNonnull() ) pdgId = abs(parton->pdgId());
    ShallowClonePtrCandidate const & matchedJet = flavHist.matchedJet();
    if ( matchedJet.masterClonePtr().isNonnull() ) {
      if ( pdgId == 4 ) nc++;
    }
  }
  return nc;
}


// 
// Calculate the maximum delta R of the highest flavor q-qbar pair
// 
// 1. loop over the history vector 
// 2. take the highest flavor q-qbar pair
// 3. calculate delta r
// 4. if more than one pair, take highest delta R
std::pair<double, int> FlavorHistoryEvent::calculateDR() const
{
  // if we already calculated this, just return the previous value
  if ( isCached() ) return std::pair<double,int>(dR_, highestFlavor_);

  // get map of (flavor,delta R max)
  map<int,double> maxDR;
  

  // Keep track of highest flavor
  int highestFlavor = -1;
  // get iterators to the history vector
  const_iterator i = histories_.begin(),
    ibegin = histories_.begin(),
    iend = histories_.end();
  // loop over history vector
  for ( ; i != iend; ++i ) {
    FlavorHistory const & flavHist = *i;
    CandidatePtr const & parton = flavHist.parton();
    // check flavor of parton... 
    int flavor = abs(parton->pdgId());
    // if we haven't seen it yet, keep traack of it
    if ( maxDR.find( flavor ) == maxDR.end() ) {
      maxDR[flavor] = -1.0;
    }
    
    // now get the two jets (matched, and sister)
    ShallowClonePtrCandidate const & matchedJet = flavHist.matchedJet();
    ShallowClonePtrCandidate const & sisterJet = flavHist.sisterJet();
    TLorentzVector p41 ( matchedJet.px(), matchedJet.py(), matchedJet.pz(), matchedJet.energy() );
    TLorentzVector p42 ( sisterJet.px(), sisterJet.py(), sisterJet.pz(), sisterJet.energy() );


    // if highest parton with jet so far, keep it
    if ( flavor > highestFlavor && matchedJet.masterClonePtr().isNonnull() ) {
      highestFlavor = flavor;
    }

    // if there is a jet, and a sister jet, get the DR and calculate max dr
    if ( matchedJet.masterClonePtr().isNonnull() && sisterJet.masterClonePtr().isNonnull() ) {
      double dr = p41.DeltaR( p42 );
      if ( dr > maxDR[flavor] ) {
	maxDR[flavor] = dr;
      }
    }
  }
  
  // now find the highest flavor max DR
  if ( highestFlavor > 0 ) {
    return std::pair<double,int>(maxDR[highestFlavor], highestFlavor);
  } else {
    return std::pair<double,int>(-1.0, 0);
  }
  
}
