#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistorySelectorUtil.h"

#include <iostream>

using namespace std;
using namespace reco;

FlavorHistorySelectorUtil::FlavorHistorySelectorUtil( unsigned int flavor,
						      unsigned int noutput,
						      flavor_vector const & flavorSource,
						      double minDR,
						      double maxDR,
						      bool verbose ) :
  flavor_(flavor),
  noutput_(noutput),
  flavorSource_(flavorSource),
  minDR_(minDR),
  maxDR_(maxDR),
  verbose_(verbose)
{
  
  // Deal with the case if minDR == maxDR, just increment maxDR by epsilon
  if ( maxDR_ - minDR_ <= 0.001 ) maxDR_ += 0.001;
}

bool FlavorHistorySelectorUtil::select(unsigned int nb,
				       unsigned int nc,
				       unsigned int highestFlavor,
				       FlavorHistory::FLAVOR_T flavorSource,
				       double dr ) const 
{
  
  // Print out some information about this event
  if ( verbose_ ) {
    cout << "Looking at flavor history event: " << endl;
    cout << "source   = " << flavorSource << endl;
    cout << "nbjet    = " << nb << endl;
    cout << "ncjet    = " << nc << endl;
    cout << "flavor   = " << highestFlavor << endl;
    cout << "dr       = " << dr << endl;
  }

  // First check that the highest flavor in the event is what this
  // filter is checking. Otherwise we need to fail the event,
  // since it should be handled by another filter
  if ( highestFlavor > static_cast<unsigned int>(flavor_) ) {
    if ( verbose_ ) cout << "Rejecting event, highest flavor is " << highestFlavor << endl;
    return false;
  }

  // Next check that the flavor source is one of the desired ones
  vector<int>::const_iterator iflavorSource = find( flavorSource_.begin(), flavorSource_.end(), static_cast<int>(flavorSource) );
  if ( iflavorSource == flavorSource_.end() ) {
    if ( verbose_ ) cout << "Rejecting event, didn't find flavor source " << static_cast<int>(flavorSource) << endl;
    return false;
  }
  
  // If we are examining b quarks
  if ( flavor_ == reco::FlavorHistory::bQuarkId ) {
    // if we have no b quarks, return false
    if ( nb <= 0 ) {
      if ( verbose_ ) cout << "Rejecting event, nb = " << nb << endl;
      return false;
    }
    // here, nb > 0
    else {
      // if we want 1 b, require nb == 1
      if ( noutput_ == 1 && nb == 1 ) {
	if ( verbose_ ) cout << "Accepting event" << endl;
	return true;
      }
      // if we want 2 b, then look at delta R
      else if ( noutput_ > 1 && nb > 1 ) {
	// If dr is within the range we want, pass.
	// Otherwise, fail.
	if ( verbose_ ) cout << "Want multiples, dr = " << dr << endl;
	return ( dr >= minDR_ && dr < maxDR_ );
      }
      // otherwise return false
      else {
	if ( verbose_ ) cout << "Rejecting event, isn't output = 1 + nb = 1, or output > 0 and delta R in proper range" << endl;
	return false;
      }
    }// end if nb > 0
    
  } // end if flavor is b quark

  // If we are examining c quarks
  else if ( flavor_ == reco::FlavorHistory::cQuarkId ) {
    // make sure there are no b quarks in the event.
    // If there are, another filter should handle it.
    if ( nb > 0 ) return false;
    
    // if we have no c quarks, return false
    if ( nc <= 0 ) return false;
    // here, nc > 0
    else {
      // if we want 1 c, require nc == 1
      if ( noutput_ == 1 && nc == 1 ) {
	return true;
      }
      // if we want 2 c, then look at delta R
      else if ( noutput_ > 1 && nc > 1 ) {
	// If dr is within the range we want, pass.
	// Otherwise, fail.
	return ( dr >= minDR_ && dr < maxDR_ );
      }
      // otherwise return false
      else {
	return false;
      }
    }// end if nc > 0
    
  }
  // Otherwise return false
  else {
    if ( verbose_ ) cout << "Something is weird, flavor is " << flavor_ << endl;
    return false;
  }
}
