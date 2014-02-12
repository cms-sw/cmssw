#include "LocalMaximum2DSeedFinder.h"

const std::vector<unsigned> LocalMaximum2DSeedFinder::_noNeighbours;

// the starting state of seedable is all false!
void LocalMaximum2DSeedFinder::
findSeeds( const edm::Handle<reco::PFRecHitCollection>& input,
	   const std::vector<bool>& mask,
	   std::vector<bool>& seedable ) {
  std::vector<bool> usable(true,input->size());
  for( unsigned idx = 0; idx < input->size(); ++idx ) {    
    if( !mask[idx] ) continue; // cannot seed masked objects
    const reco::PFRecHit& maybeseed = input->at(idx);
    if( maybeseed.energy() < _seedingThreshold || 
	maybeseed.pt2() < _seedingThresholdPt2   ) usable[idx] = false;      
    if( !usable[idx] ) continue;
    //get the neighbours of this seed
    const std::vector<unsigned>* myNeighbours;
    switch( _nNeighbours ) {
    case 0: // for HF clustering
      myNeighbours = &_noNeighbours;
      break;
    case 4:
      myNeighbours = &maybeseed.neighbours4();
      break;
    case 8:
      myNeighbours = &maybeseed.neighbours8();
      break;
    default:
      throw cms::Exception("InvalidConfiguration")
	<< "LocalMaximum2DSeedFinder only accepts nNeighbors = {0,4,8}";    
    }
    seedable[idx] = true;
    for( const unsigned neighbour : *myNeighbours ) {
      if( !mask[neighbour] ) continue;
      if( input->at(neighbour).energy() > maybeseed.energy() ) {
	seedable[idx] = false;
	break;
      }
    }
    if( seedable[idx] ) {
      for( const unsigned neighbour : *myNeighbours ) {
	usable[neighbour] = false;
      }
    }
  }
}
