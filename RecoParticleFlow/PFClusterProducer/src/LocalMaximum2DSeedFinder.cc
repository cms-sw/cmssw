#include "LocalMaximum2DSeedFinder.h"

const std::vector<unsigned> LocalMaximum2DSeedFinder::_noNeighbours;

namespace {
  bool greaterByEnergy(const std::pair<unsigned,double>& a,
		       const std::pair<unsigned,double>& b) {
    return a.second > b.second;
  }
}

// the starting state of seedable is all false!
void LocalMaximum2DSeedFinder::
findSeeds( const edm::Handle<reco::PFRecHitCollection>& input,
	   const std::vector<bool>& mask,
	   std::vector<bool>& seedable ) {
  std::vector<bool> usable(input->size(),true);
  //need to run over energy sorted rechits
  std::vector<std::pair<unsigned,double> > ordered_hits;
  ordered_hits.reserve(input->size());
  for( unsigned i = 0; i < input->size(); ++i ) {
    std::pair<unsigned,double> val = std::make_pair(i,input->at(i).energy());
    auto pos = std::lower_bound(ordered_hits.begin(),ordered_hits.end(),
				val, greaterByEnergy);
    ordered_hits.insert(pos,val);
  }  
  
  for( const auto& idx_e : ordered_hits ) {    
    const unsigned idx = idx_e.first;
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
