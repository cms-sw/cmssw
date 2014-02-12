#include "Basic2DGenericPFlowClusterizer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

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

void Basic2DGenericPFlowClusterizer::
buildPFClusters(const reco::PFClusterCollection& input,
		const std::vector<bool>& seedable,
		reco::PFClusterCollection& output) {
  reco::PFClusterCollection clustersInTopo;
  for( const auto& topocluster : input ) {
    clustersInTopo.clear();
    seedPFClustersFromTopo(topocluster,seedable,clustersInTopo);
    const unsigned tolScal = 
      std::pow(std::max(1.0,clustersInTopo.size()-1.0),2.0);
    growPFClusters(topocluster,seedable,tolScal,0,tolScal,clustersInTopo);
    prunePFClusters(clustersInTopo);
    if( clustersInTopo.size() == 1 ) {
      _allCellsPosCalc->calculateAndSetPosition(clustersInTopo.back());
    } else {
      _positionCalc->calculateAndSetPositions(clustersInTopo);
    }
  }
}

void Basic2DGenericPFlowClusterizer::
seedPFClustersFromTopo(const reco::PFCluster& topo,
		       const std::vector<bool>& seedable,
		       reco::PFClusterCollection& initialPFClusters) const {
  const auto& recHitFractions = topo.recHitFractions();
  for( const auto& rhf : recHitFractions ) {
    if( !seedable[rhf.recHitRef().key()] ) continue;
    initialPFClusters.push_back(reco::PFCluster());
    reco::PFCluster& current = initialPFClusters.back();
    current.addRecHitFraction(rhf);
    current.setSeed(rhf.recHitRef()->detId());
    _positionCalc->calculateAndSetPosition(current);
  }
}

void Basic2DGenericPFlowClusterizer::
growPFClusters(const reco::PFCluster& topo,
	       const std::vector<bool>& seedable,
	       const unsigned toleranceScaling,
	       const unsigned iter,
	       double diff,
	       reco::PFClusterCollection& clusters) const {
  if( iter >= _maxIterations ) {
    LOGDRESSED("Basic2DGenericPFlowClusterizer:growAndStabilizePFClusters")
      <<"reached " << _maxIterations << " iterations, terminated position "
      << "fit with diff = " << diff;
  }    
  if( iter >= _maxIterations || 
      diff <= _stoppingTolerance*toleranceScaling) return;
  // store the previous iteration
  reco::PFClusterCollection clustersLastIter = std::move(clusters);
  clusters.clear();
  clusters.reserve(clustersLastIter.size());
  // loop over topo cluster and grow current PFCluster hypothesis 
  std::vector<double> dist, frac;
  double fractot = 0, fraction = 0;
  for( const reco::PFRecHitFraction& rhf : topo.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();
    const math::XYZPoint& topocellpos_xyz = refhit->position();
    dist.clear(); frac.clear(); fractot = 0;
    // add rechits to clusters, calculating fraction based on distance
    for( const auto& clusterLastIter : clustersLastIter ) {
      const math::XYZPoint& clusterpos_xyz = clusterLastIter.position();
      fraction = 0.0;
      math::XYZVector deltav = clusterpos_xyz - topocellpos_xyz;
      const double d = deltav.R()/_showerSigma;
      dist.push_back( d );
      if( d > 10 ) {
	LOGDRESSED("Basic2DGenericPFlowClusterizer:growAndStabilizePFClusters")
	  << "Warning! :: pfcluster-topocell distance is too large! d= "
	  << d;
      }
      // fraction assignment logic
      if( refhit->detId() == clusterLastIter.seed() && _excludeOtherSeeds ) {
	fraction = 1.0;	
      } else if ( seedable[refhit.key()] && _excludeOtherSeeds ) {
	fraction = 0.0;
      } else {
	fraction = clusterLastIter.energy() * std::exp( -d*d/2.0 );
      }
      fractot += fraction;
      frac.push_back(fraction);
    }
    for( unsigned i = 0; i < clustersLastIter.size(); ++i ) {
      clusters.push_back( clustersLastIter[i] );
      if( fractot > 0.0 ) frac[i]/=fractot;
      else continue;
      // if the fraction has been set to 0, the cell 
      // is now added to the cluster - careful ! (PJ, 19/07/08)
      // BUT KEEP ONLY CLOSE CELLS OTHERWISE MEMORY JUST EXPLOSES
      // (PJ, 15/09/08 <- similar to what existed before the 
      // previous bug fix, but keeps the close seeds inside, 
      // even if their fraction was set to zero.)
      // Also add a protection to keep the seed in the cluster 
      // when the latter gets far from the former. These cases
      // (about 1% of the clusters) need to be studied, as 
      // they create fake photons, in general.
      // (PJ, 16/09/08) 
      if( dist[i] < 10.0 || frac[i] > 0.9999 ) {	
	clusters[i].addRecHitFraction(reco::PFRecHitFraction(refhit, frac[i]));
      }
    }
  }
  // recalculate positions and calculate convergence parameter
  double diff2 = 0.0;
  for( unsigned i = 0; i < clustersLastIter.size(); ++i ) {
    if( clustersLastIter.size() == 1 ) {
      _allCellsPosCalc->calculateAndSetPosition(clusters[i]);
    } else {
      _positionCalc->calculateAndSetPosition(clusters[i]);
    }
    const double delta2 = 
      reco::deltaR2(clusters[i].positionREP(),
		    clustersLastIter[i].positionREP());
    if( delta2 > diff2 ) diff2 = delta2;
  }
  diff = std::sqrt(diff2);
  dist.clear(); frac.clear(); clustersLastIter.clear(); // avoid badness
  growPFClusters(topo,seedable,toleranceScaling,iter+1,diff,clusters);
}

void Basic2DGenericPFlowClusterizer::
prunePFClusters(reco::PFClusterCollection& clusters) const {
  for( auto& cluster : clusters ) {
    std::vector<reco::PFRecHitFraction>& allFracs = cluster.recHitFractions();
    std::vector<reco::PFRecHitFraction> prunedFracs;
    prunedFracs.reserve(cluster.recHitFractions().size());
    for( const auto& frac : allFracs ) {
      if( frac.fraction() > _minFractionToKeep ) prunedFracs.push_back(frac);
    }
    prunedFracs.shrink_to_fit();
    allFracs.clear();    
    allFracs = std::move(prunedFracs);
  }
}
