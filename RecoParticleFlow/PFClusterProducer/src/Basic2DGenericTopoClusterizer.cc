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
buildTopoClusters(const edm::Handle<reco::PFRecHitCollection>& input,
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
    auto pos = std::lower_bound(seeds.begin(),seeds.end(),val,greaterByEnergy);
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
  if( cell->energy() < _gatheringThreshold || 
      cell->pt2() < _gatheringThresholdPt2 ) {
    LOGDRESSED("GenericTopoCluster::buildTopoCluster()")
      << "RecHit " << cell->detId() << " with enegy " 
      << cell->energy() << " GeV was rejected!." << std::endl;
    return;
  }

  used[cell.key()] = true;
  topocluster.addRecHitFraction(reco::PFRecHitFraction(cell, 1.0));
  
  const std::vector<unsigned>& neighbors = 
    _useCornerCells ? cell->neighbours8() : cell->neighbours4();

  for( unsigned idx : neighbors ) {
    if( used[idx] || !rechitMask[idx] ) {
      LOGDRESSED("GenericTopoCluster::buildTopoCluster()")
      	<< "  RecHit " << cell->detId() << "\'s" 
	<< " neighbor RecHit " << input->at(idx).detId() << " with enegy " 
	<< input->at(idx).energy() << " GeV was rejected!" 
	<< " Reasons : " << used[idx] << " (used) " 
	<< !rechitMask[idx] << " (masked)." << std::endl;
      continue;
    }
    buildTopoCluster(input,rechitMask,makeRefhit(input,idx),used,topocluster);
  }
}
