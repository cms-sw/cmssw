#include "Basic2DGenericTopoClusterizer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

namespace {
  bool greaterByEnergy(const std::pair<unsigned,double>& a,
		       const std::pair<unsigned,double>& b) {
    return a.second > b.second;
  }
}

void Basic2DGenericTopoClusterizer::
buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
	      const std::vector<bool>& rechitMask,
	      const std::vector<bool>& seedable,
	      reco::PFClusterCollection& output) {  
  std::vector<bool> used(input->size(),false);
  std::vector<std::pair<unsigned,double> > seeds;
  
  // get the seeds and sort them descending in energy
  seeds.reserve(input->size());  
  for( unsigned i = 0; i < input->size(); ++i ) {
    if( !rechitMask[i] || !seedable[i] || used[i] ) continue;
    std::pair<unsigned,double> val = std::make_pair(i,input->at(i).energy());
    auto pos = std::upper_bound(seeds.begin(),seeds.end(),val,greaterByEnergy);
    seeds.insert(pos,val);
  }
  
  reco::PFCluster temp;
  for( const auto& idx_e : seeds ) {    
    const int seed = idx_e.first;
    if( !rechitMask[seed] || !seedable[seed] || used[seed] ) continue;    
    temp.reset();
    buildTopoCluster(input,rechitMask,makeRefhit(input,seed),used,temp);
    if( temp.recHitFractions().size() ) output.push_back(temp);
  }
}

void Basic2DGenericTopoClusterizer::
buildTopoCluster(const edm::Handle<reco::PFRecHitCollection>& input,
		 const std::vector<bool>& rechitMask,
		 const reco::PFRecHitRef& cell,
		 std::vector<bool>& used,		 
		 reco::PFCluster& topocluster) {
  int cell_layer = (int)cell->layer();
  if( cell_layer == PFLayer::HCAL_BARREL2 && 
      std::abs(cell->positionREP().eta()) > 0.34 ) {
      cell_layer *= 100;
    }    
  const std::pair<double,double>& thresholds =
      _thresholds.find(cell_layer)->second;
  if( cell->energy() < thresholds.first || 
      cell->pt2() < thresholds.second ) {
    LOGDRESSED("GenericTopoCluster::buildTopoCluster()")
      << "RecHit " << cell->detId() << " with enegy " 
      << cell->energy() << " GeV was rejected!." << std::endl;
    return;
  }

  used[cell.key()] = true;
  topocluster.addRecHitFraction(reco::PFRecHitFraction(cell, 1.0));
  
  const reco::PFRecHitRefVector& neighbours = 
    ( _useCornerCells ? cell->neighbours8() : cell->neighbours4() );
  
  for( const reco::PFRecHitRef nb : neighbours ) {
    if( used[nb.key()] || !rechitMask[nb.key()] ) {
      LOGDRESSED("GenericTopoCluster::buildTopoCluster()")
      	<< "  RecHit " << cell->detId() << "\'s" 
	<< " neighbor RecHit " << input->at(nb.key()).detId() 
	<< " with enegy " 
	<< input->at(nb.key()).energy() << " GeV was rejected!" 
	<< " Reasons : " << used[nb.key()] << " (used) " 
	<< !rechitMask[nb.key()] << " (masked)." << std::endl;
      continue;
    }
    buildTopoCluster(input,rechitMask,nb,used,topocluster);
  }
}
