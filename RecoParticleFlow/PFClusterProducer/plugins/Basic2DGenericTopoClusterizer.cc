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
buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
	      const std::vector<bool>& rechitMask,
	      const std::vector<bool>& seedable,
	      reco::PFClusterCollection& output) {
  auto const & hits = *input;  
  std::vector<bool> used(hits.size(),false);
  std::vector<unsigned int> seeds;
  
  // get the seeds and sort them descending in energy
  seeds.reserve(hits.size());  
  for( unsigned int i = 0; i < hits.size(); ++i ) {
    if( !rechitMask[i] || !seedable[i] || used[i] ) continue;
    seeds.emplace_back(i);
  }
  // maxHeap would be better
  std::sort(seeds.begin(),seeds.end(),
            [&](unsigned int i, unsigned int j) { return hits[i].energy()>hits[j].energy();});  
  
  reco::PFCluster temp;
  for( auto seed : seeds ) {    
    if( !rechitMask[seed] || !seedable[seed] || used[seed] ) continue;    
    temp.reset();
    buildTopoCluster(input,rechitMask,seed,used,temp);
    if( !temp.recHitFractions().empty() ) output.push_back(temp);
  }
}

void Basic2DGenericTopoClusterizer::
buildTopoCluster(const edm::Handle<reco::PFRecHitCollection>& input,
		 const std::vector<bool>& rechitMask,
		 unsigned int kcell,
		 std::vector<bool>& used,		 
		 reco::PFCluster& topocluster) {
  auto const & cell = (*input)[kcell];
  int cell_layer = (int)cell.layer();
  if( cell_layer == PFLayer::HCAL_BARREL2 && 
      std::abs(cell.positionREP().eta()) > 0.34 ) {
      cell_layer *= 100;
    }    

  std::tuple<std::vector<int> ,std::vector<double> , std::vector<double> > thresholds = _thresholds.find(cell_layer)->second;

  for (unsigned int j=0; j<(std::get<1>(thresholds)).size(); ++j) {
    if((cell_layer == PFLayer::HCAL_BARREL1 || cell_layer == PFLayer::HCAL_ENDCAP) && (cell.depth()!=std::get<0>(thresholds)[j])) continue;

    if( cell.energy() < std::get<1>(thresholds)[j] ||
	cell.pt2() < std::get<2>(thresholds)[j]  ) {
      LOGDRESSED("GenericTopoCluster::buildTopoCluster()")
	<< "RecHit " << cell.detId() << " with enegy "
	<< cell.energy() << " GeV was rejected!." << std::endl;
      return;
    }

  }


  auto k = kcell;
  used[k] = true;
  auto ref = makeRefhit(input,k);
  topocluster.addRecHitFraction(reco::PFRecHitFraction(ref, 1.0));
  
  auto const & neighbours = 
    ( _useCornerCells ? cell.neighbours8() : cell.neighbours4() );
  
  for( auto nb : neighbours ) {
    if( used[nb] || !rechitMask[nb] ) {
      LOGDRESSED("GenericTopoCluster::buildTopoCluster()")
      	<< "  RecHit " << cell.detId() << "\'s" 
	<< " neighbor RecHit " << input->at(nb).detId() 
	<< " with enegy " 
	<< input->at(nb).energy() << " GeV was rejected!" 
	<< " Reasons : " << used[nb] << " (used) " 
	<< !rechitMask[nb] << " (masked)." << std::endl;
      continue;
    }
    buildTopoCluster(input,rechitMask,nb,used,topocluster);
  }
}
