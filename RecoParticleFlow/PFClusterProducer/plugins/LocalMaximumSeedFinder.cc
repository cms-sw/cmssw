#include "LocalMaximumSeedFinder.h"

#include <algorithm>
#include <queue>
#include <cfloat>
#include "CommonTools/Utils/interface/DynArray.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace {
  const reco::PFRecHit::Neighbours  _noNeighbours(nullptr,0);
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
	      {"HCAL_BARREL2_RING1", 19}, 
	      {"HCAL_ENDCAP",(int)PFLayer::HCAL_ENDCAP},
	      {"HF_EM",(int)PFLayer::HF_EM},
	      {"HF_HAD",(int)PFLayer::HF_HAD} }) {
  const std::vector<edm::ParameterSet>& thresholds =
    conf.getParameterSetVector("thresholdsByDetector");
  for( const auto& pset : thresholds ) {
    const std::string& det = pset.getParameter<std::string>("detector");

    std::vector<int> depths;
    std::vector<double> thresh_E;
    std::vector<double> thresh_pT ;
    std::vector<double> thresh_pT2;

    if (det==std::string("HCAL_BARREL1") || det==std::string("HCAL_ENDCAP")) {
      depths = pset.getParameter<std::vector<int> >("depths");
      thresh_E = pset.getParameter<std::vector<double> >("seedingThreshold");
      thresh_pT = pset.getParameter<std::vector<double> >("seedingThresholdPt");
      if(thresh_E.size()!=depths.size() || thresh_pT.size()!=depths.size()) {
	throw cms::Exception("InvalidGatheringThreshold")
	  << "gatheringThresholds mismatch with the numbers of depths";
      }
    } else {
      depths.push_back(0);
      thresh_E.push_back(pset.getParameter<double>("seedingThreshold"));
      thresh_pT.push_back(pset.getParameter<double>("seedingThresholdPt"));
    }

    for(unsigned int i=0;i < thresh_pT.size();++i){
      thresh_pT2.push_back(thresh_pT[i]*thresh_pT[i]);
    }

    auto entry = _layerMap.find(det);
    if( entry == _layerMap.end() ) {
      throw cms::Exception("InvalidDetectorLayer")
	<< "Detector layer : " << det << " is not in the list of recognized"
	<< " detector layers!";
    }

    _thresholds[entry->second+layerOffset]= 
                       std::make_tuple(depths,thresh_E,thresh_pT2);
  }
}

// the starting state of seedable is all false!
void LocalMaximumSeedFinder::
findSeeds( const edm::Handle<reco::PFRecHitCollection>& input,
	   const std::vector<bool>& mask,
	   std::vector<bool>& seedable ) {

  auto nhits = input->size();
  initDynArray(bool,nhits,usable,true);
  //need to run over energy sorted rechits
  declareDynArray(float,nhits,energies);
  unInitDynArray(int,nhits,qst); // queue storage
  auto cmp = [&](int i, int j) { return energies[i] < energies[j]; };
  std::priority_queue<int, DynArray<int>, decltype(cmp)> ordered_hits(cmp,std::move(qst));

  for( unsigned i = 0; i < nhits; ++i ) {
    if( !mask[i] ) continue; // cannot seed masked objects
    auto const & maybeseed = (*input)[i];
    energies[i]=maybeseed.energy();
    int seedlayer = (int)maybeseed.layer();
    if( seedlayer == PFLayer::HCAL_BARREL2 &&
        std::abs(maybeseed.positionREP().eta()) > 0.34 ) {
      seedlayer = 19;
    }
    auto const & thresholds = _thresholds[seedlayer+layerOffset];

    for (unsigned int j=0; j<(std::get<1>(thresholds)).size(); ++j) {
      if((seedlayer == PFLayer::HCAL_BARREL1 || seedlayer == PFLayer::HCAL_ENDCAP) && (maybeseed.depth()!=std::get<0>(thresholds)[j])) continue;

	if( maybeseed.energy() < std::get<1>(thresholds)[j] ||
	    maybeseed.pt2() < std::get<2>(thresholds)[j]   ) usable[i] = false;
	if( !usable[i] ) continue;
	ordered_hits.push(i);

    }
  }


  while(!ordered_hits.empty() ) {
    auto idx = ordered_hits.top();
    ordered_hits.pop();  
    if( !usable[idx] ) continue;
    //get the neighbours of this seed
    auto const & maybeseed = (*input)[idx];
    reco::PFRecHit::Neighbours  myNeighbours;
    switch( _nNeighbours ) {
    case -1:
      myNeighbours = maybeseed.neighbours();
      break;
    case 0: // for HF clustering
      myNeighbours = _noNeighbours;
      break;
    case 4:
      myNeighbours = maybeseed.neighbours4();
      break;
    case 8:
      myNeighbours = maybeseed.neighbours8();
      break;
    default:
      throw cms::Exception("InvalidConfiguration")
	<< "LocalMaximumSeedFinder only accepts nNeighbors = {-1,0,4,8}";    
    }
    seedable[idx] = true;
    for( auto neighbour : myNeighbours ) {
      if( !mask[neighbour] ) continue;
      if( energies[neighbour] > energies[idx] ) {
//        std::cout << "how this can be?" << std::endl;
	seedable[idx] = false;	
	break;
      }
    }
    if( seedable[idx] ) {
      for( auto neighbour : myNeighbours ) {
	usable[neighbour] = false;
      }
    }
  }

  LogDebug("LocalMaximumSeedFinder") << " found " << std::count(seedable.begin(),seedable.end(),true) << " seeds";

}
