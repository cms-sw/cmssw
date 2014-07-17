#include "SimpleArborClusterizer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterProducer/interface/Arbor.hh"

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

void SimpleArborClusterizer::
buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
	      const std::vector<bool>& rechitMask,
	      const std::vector<bool>& seedable,
	      reco::PFClusterCollection& output) {  
  const reco::PFRecHitCollection& hits = *input;
  std::vector<TVector3> arbor_points;
  std::vector<unsigned> hits_for_arbor;
  arbor::branchcoll branches;  

  // get the seeds and sort them descending in energy
  arbor_points.reserve(hits.size());
  hits_for_arbor.reserve(hits.size());  
  for( unsigned i = 0; i < hits.size(); ++i ) {
    if( !rechitMask[i] ) continue;
    const math::XYZPoint& pos = hits[i].position();
    hits_for_arbor.emplace_back(i);
    arbor_points.emplace_back(10*pos.x(),10*pos.y(),10*pos.z());
  }

  branches = arbor::Arbor(arbor_points,_cellSize,_layerThickness,_distSeedForMerge);
  output.reserve(branches.size());
  
  for( auto& branch : branches ) {
    if( _killNoiseClusters && branch.size() <= _maxNoiseClusterSize ) {
      continue;
    }
    // sort hits by radius
    std::sort(branch.begin(),branch.end(),
	      [&](const arbor::branch::value_type& a,
		  const arbor::branch::value_type& b) {
		return ( hits[hits_for_arbor[a]].position().Mag2() <
			 hits[hits_for_arbor[b]].position().Mag2()   );
	      });
    const reco::PFRecHit& inner_hit = hits[hits_for_arbor[branch[0]]];
    PFLayer::Layer inner_layer = inner_hit.layer();
    const math::XYZPoint& inner_pos = inner_hit.position();    
    output.emplace_back(inner_layer,branch.size(),
			inner_pos.x(),inner_pos.y(),inner_pos.z());
    reco::PFCluster& current = output.back();
    current.setSeed(inner_hit.detId());
    for( const auto& hit : branch ) {
      const reco::PFRecHitRef& refhit = 
	reco::PFRecHitRef(input,hits_for_arbor[hit]);
      current.addRecHitFraction(reco::PFRecHitFraction(refhit,1.0));
    }
    LogDebug("SimpleArborClusterizer")
      << "Made cluster: " << current << std::endl;
  }
}
