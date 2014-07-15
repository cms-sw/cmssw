#include "LocalMaximumSeedFinder.h"

const reco::PFRecHitRefVector LocalMaximumSeedFinder::_noNeighbours;

namespace {
  bool greaterByEnergy(const std::pair<unsigned,double>& a,
		       const std::pair<unsigned,double>& b) {
    return a.second > b.second;
  }
}

LocalMaximumSeedFinder::
LocalMaximumSeedFinder(const edm::ParameterSet& conf) : 
  SeedFinderBase(conf),   
  _nNeighbours(conf.getParameter<int>("nNeighbours")),
  _layerMap({ {"PS2",(int)PFLayer::PS2},
	      {"PS1",(int)PFLayer::PS1},
	      {"ECAL_ENDCAP",(int)PFLayer::ECAL_ENDCAP},
	      {"ECAL_BARREL",(int)PFLayer::ECAL_BARREL},
	      {"NONE",(int)PFLayer::NONE},
	      {"HCAL_BARREL1",(int)PFLayer::HCAL_BARREL1},
	      {"HCAL_BARREL2_RING0",(int)PFLayer::HCAL_BARREL2},
              // hack to deal with ring1 in HO 
	      {"HCAL_BARREL2_RING1",100*(int)PFLayer::HCAL_BARREL2}, 
	      {"HCAL_ENDCAP",(int)PFLayer::HCAL_ENDCAP},
	      {"HF_EM",(int)PFLayer::HF_EM},
	      {"HF_HAD",(int)PFLayer::HF_HAD} }) {
  const std::vector<edm::ParameterSet>& thresholds =
    conf.getParameterSetVector("thresholdsByDetector");
  for( const auto& pset : thresholds ) {
    const std::string& det = pset.getParameter<std::string>("detector");
    const double& thresh_E = pset.getParameter<double>("seedingThreshold");
    const double& thresh_pT = pset.getParameter<double>("seedingThresholdPt");
    const double thresh_pT2 = thresh_pT*thresh_pT;
    auto entry = _layerMap.find(det);
    if( entry == _layerMap.end() ) {
      throw cms::Exception("InvalidDetectorLayer")
	<< "Detector layer : " << det << " is not in the list of recognized"
	<< " detector layers!";
    }
    _thresholds.emplace(_layerMap.find(det)->second, 
			std::make_pair(thresh_E,thresh_pT2));
  }
}

// the starting state of seedable is all false!
void LocalMaximumSeedFinder::
findSeeds( const edm::Handle<reco::PFRecHitCollection>& input,
	   const std::vector<bool>& mask,
	   std::vector<bool>& seedable ) {
  std::vector<bool> usable(input->size(),true);
  //need to run over energy sorted rechits
  std::vector<std::pair<unsigned,double> > ordered_hits;
  ordered_hits.reserve(input->size());
  for( unsigned i = 0; i < input->size(); ++i ) {
    std::pair<unsigned,double> val = std::make_pair(i,input->at(i).energy());
    auto pos = std::upper_bound(ordered_hits.begin(),ordered_hits.end(),
				val, greaterByEnergy);
    ordered_hits.insert(pos,val);
  }      

  for( const auto& idx_e : ordered_hits ) {    
    const unsigned idx = idx_e.first;    
    if( !mask[idx] ) continue; // cannot seed masked objects
    const reco::PFRecHit& maybeseed = input->at(idx);
    int seedlayer = (int)maybeseed.layer();
    if( seedlayer == PFLayer::HCAL_BARREL2 && 
	std::abs(maybeseed.positionREP().eta()) > 0.34 ) {
      seedlayer *= 100;
    }    
    const std::pair<double,double>& thresholds =
      _thresholds.find(seedlayer)->second;
    if( maybeseed.energy() < thresholds.first || 
	maybeseed.pt2() < thresholds.second   ) usable[idx] = false;      
    if( !usable[idx] ) continue;
    //get the neighbours of this seed
    const reco::PFRecHitRefVector* myNeighbours;
    switch( _nNeighbours ) {
    case -1:
      myNeighbours = &maybeseed.neighbours();
      break;
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
	<< "LocalMaximumSeedFinder only accepts nNeighbors = {-1,0,4,8}";    
    }
    seedable[idx] = true;
    for( const reco::PFRecHitRef& neighbour : *myNeighbours ) {
      if( !mask[neighbour.key()] ) continue;
      if( neighbour->energy() > maybeseed.energy() ) {
	seedable[idx] = false;	
	break;
      }
    }
    if( seedable[idx] ) {
      for( const reco::PFRecHitRef& neighbour : *myNeighbours ) {
	usable[neighbour.key()] = false;
      }
    }
  }
}
