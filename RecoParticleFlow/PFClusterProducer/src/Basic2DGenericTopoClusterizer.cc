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

void Basic2DGenericTopoClusterizer::
buildTopoClusters(const reco::PFRecHitRefVector& input,
		  const std::vector<bool>& rechitMask,
		  const std::vector<bool>& seedable,
		  reco::PFClusterCollection& output) {  
  std::vector<bool> used(false,input.size());

  reco::PFCluster temp;
  for( const reco::PFRecHitRef& cell : input ) {
    if( !seedable[cell.key()] || used[cell.key()] ) continue;
    temp.reset();
    buildTopoCluster(input,rechitMask,used,cell,temp);
    if( temp.recHitFractions().size() ) output.push_back(temp);
  }
}

void Basic2DGenericTopoClusterizer::
buildTopoCluster(const reco::PFRecHitRefVector& input,
		 const std::vector<bool>& rechitMask,
		 std::vector<bool>& used,
		 const reco::PFRecHitRef& cell,
		 reco::PFCluster& topocluster) {
  if( cell->energy() < _gatheringThreshold || 
      cell->pt2() < _gatheringThresholdPt2 ) {
    LOGDRESSED("GenericTopoCluster::buildTopoCluster()")
      << "RecHit " << cell->detId() << " with enegy " 
      << cell->energy() << " GeV was rejected!.";
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
	<< " neighbor RecHit " << input[idx]->detId() << " with enegy " 
	<< input[idx]->energy() << " GeV was rejected!" 
	<< " Reasons : " << used[idx] << " (used) " 
	<< !rechitMask[idx] << " (masked)." ;
      continue;
    }
    buildTopoCluster(input,rechitMask,used,input[idx],topocluster);
  }
}
