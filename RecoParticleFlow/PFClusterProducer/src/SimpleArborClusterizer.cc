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
  std::vector<std::pair<TVector3,float> > arbor_points[2];
  std::vector<unsigned> hits_for_arbor[2];
  arbor::branchcoll the_branches[2];  

  // get the seeds and sort them descending in energy
  arbor_points[0].reserve(hits.size()/2);
  arbor_points[1].reserve(hits.size()/2);
  hits_for_arbor[0].reserve(hits.size()/2);  
  hits_for_arbor[1].reserve(hits.size()/2);
  for( unsigned i = 0; i < hits.size(); ++i ) {
    if( !rechitMask[i] ) continue;
    const math::XYZPoint& pos = hits[i].position();     
    TVector3 v3pos(10.0*pos.x(),10.0*pos.y(),10.0*pos.z()); // convert to mm
    if( v3pos.z() < 0 ) {
      hits_for_arbor[0].emplace_back(i);
      arbor_points[0].emplace_back(v3pos,hits[i].energy());
    } else {
      hits_for_arbor[1].emplace_back(i);
      arbor_points[1].emplace_back(v3pos,hits[i].energy());
    }
  }
  edm::LogInfo("ArborProgress") 
    << "arbor loaded: " << arbor_points[0].size() << " in negative endcap!";
  edm::LogInfo("ArborProgress") 
    << "arbor loaded: " << arbor_points[1].size() << " in positive endcap!";
  

  the_branches[0] = arbor::Arbor(arbor_points[0],_cellSize,_layerThickness,_distSeedForMerge,_allowSameLayerSeedMerge);
  edm::LogInfo("ArborProgress") << "arbor clustered negative endcap!";
  the_branches[1] = arbor::Arbor(arbor_points[1],_cellSize,_layerThickness,_distSeedForMerge,_allowSameLayerSeedMerge);
  edm::LogInfo("ArborProgress") << "arbor clustered positive endcap!";
  output.reserve(the_branches[0].size()+the_branches[1].size());
  
  for( unsigned iside = 0; iside < 2; ++iside ) {
    arbor::branchcoll& branches = the_branches[iside];
    for( auto& branch : branches ) {
      if( _killNoiseClusters && branch.size() <= _maxNoiseClusterSize ) {
	continue;
      }
      // sort hits by radius
      std::sort(branch.begin(),branch.end(),
		[&](const arbor::branch::value_type& a,
		    const arbor::branch::value_type& b) {
		  return ( hits[hits_for_arbor[iside][a]].position().Mag2() <
			   hits[hits_for_arbor[iside][b]].position().Mag2()   );
		});
      const reco::PFRecHit& inner_hit = hits[hits_for_arbor[iside][branch[0]]];
      PFLayer::Layer inner_layer = inner_hit.layer();
      const math::XYZPoint& inner_pos = inner_hit.position();    
      output.emplace_back(inner_layer,branch.size(),
			  inner_pos.x(),inner_pos.y(),inner_pos.z());
      reco::PFCluster& current = output.back();
      current.setSeed(inner_hit.detId());
      for( const auto& hit : branch ) {
	const reco::PFRecHitRef& refhit = 
	  reco::PFRecHitRef(input,hits_for_arbor[iside][hit]);
	current.addRecHitFraction(reco::PFRecHitFraction(refhit,1.0));
      }
      LogDebug("SimpleArborClusterizer")
	<< "Made cluster: " << current << std::endl;
    }
  }
  edm::LogError("ArborInfo") << "Made " << output.size() << " clusters!";
}
