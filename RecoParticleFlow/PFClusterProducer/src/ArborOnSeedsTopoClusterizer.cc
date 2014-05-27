#include "ArborOnSeedsTopoClusterizer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <unordered_map>

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

void ArborOnSeedsTopoClusterizer::
buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
	      const std::vector<bool>& rechitMask,
	      const std::vector<bool>& seedable,
	      reco::PFClusterCollection& output) {  
  std::vector<bool> used(input->size(),false);
  std::vector<std::pair<unsigned,double> > seeds;
  const reco::PFRecHitCollection& inp = *input;
  reco::PFClusterCollection topoclusters;
  // get the seeds and sort them descending in energy
  seeds.reserve(inp.size());  
  for( unsigned i = 0; i < inp.size(); ++i ) {
    if( !rechitMask[i] || !seedable[i] || used[i] ) continue;
    std::pair<unsigned,double> val = std::make_pair(i,inp.at(i).energy());
    auto pos = std::upper_bound(seeds.begin(),seeds.end(),val,greaterByEnergy);
    seeds.insert(pos,val);
  } 

  // run arbor on seeds so that we can link topoclusters
  arbor::branchcoll branches; 
  branches.reserve(seeds.size());
  arborizeSeeds(inp,seeds,branches);

  // now we build the simple topological clusters in each layer  
  reco::PFCluster temp;
  for( const auto& idx_e : seeds ) {    
    const int seed = idx_e.first;
    if( !rechitMask[seed] || !seedable[seed] || used[seed] ) continue;    
    temp.reset();
    buildTopoCluster(input,rechitMask,makeRefhit(input,seed),used,temp);
    if( temp.recHitFractions().size() ) topoclusters.push_back(temp);
  }
  
  // connect together the per-layer topo clusters with the arbor
  // branches [indiuces of arbor-branches] -> [indices of topo clusters]
  std::unordered_multimap<unsigned,unsigned> branches_to_topos;
  std::unordered_multimap<unsigned,unsigned> topos_to_branches;
  for( unsigned i = 0 ; i < topoclusters.size(); ++i  ) {
    for( const auto& rhf : topoclusters[i].recHitFractions() ) {
      if( !seedable[rhf.recHitRef().key()] ) continue;
      for( unsigned j = 0; j < branches.size(); ++j ) {
	auto ihit = std::find_if(branches[j].begin(),branches[j].end(),
				 [&](const arbor::branch::value_type& hit){
				   if( seeds[hit].first == 
				       rhf.recHitRef().key() ) return true;
				   return false;
				 });
	if( ihit != branches[j].end() ) {
	  topos_to_branches.emplace(i,j);	  
	  branches_to_topos.emplace(j,i);
	}
      }
    }
  }
  // have relational maps of topo clusters grouped together and 
  // constituent branches (shitty code and probably very slow right now)
  std::vector<std::vector<unsigned> > grouped_topos; // same indices
  std::vector<std::vector<unsigned> > grouped_branches; // same indices
  std::vector<bool> used_topo(false,topoclusters.size());
  for(unsigned i = 0; i < topoclusters.size(); ++i ) {
    if( used_topo[i] ) continue; //skip topo clusters already used
    grouped_topos.push_back(std::vector<unsigned>());
    getLinkedTopoClusters(topos_to_branches,
			  branches_to_topos,
			  topoclusters,
			  i,
			  used_topo,
			  grouped_topos.back());
  }
  for( unsigned i = 0 ; i < grouped_topos.size(); ++i ) {
    grouped_branches.push_back(std::vector<unsigned>());
    std::vector<unsigned>& current = grouped_branches.back();
    for( unsigned itopo : grouped_topos[i] ) {
      auto branch_range = topos_to_branches.equal_range(itopo);
      for( auto ib = branch_range.first; ib != branch_range.second; ++ib ) {
	auto branchid = std::find(current.begin(),current.end(),ib->second);
	if( branchid == current.end() ) {
	  current.push_back(*branchid);
	}
      }
    }
  }
  
  /*
  // next create the PFClusters from each Arborized seed group  
  for( const auto& seed : seeds ) {
    output.push_back(reco::PFCluster());
    reco::PFCluster& current = output.back();
    reco::PFRecHitFraction rhf(makeRefhit(input,seed[0]),1.0)
    current.addRecHitFraction(rhf);
    current.setSeed(inp.at[seed[0]].detId());
    for( auto rhit = seed.begin()+1; rhit != seed.end(); ++rhit ) {
      reco::PFRecHitFraction rhf2(makeRefhit(input,*rhit),1.0);
      current.addRecHitFraction(rhf2);
    }
    // calculate seeded position
    // ~~~magic~~~ (to be replaced soon)
  }

  // run semi-3D (2D position fit with center-per-layer given by 3D line fit) 
  // pf cluster position fit, loop on seeds
  const unsigned tolScal = //take log to make scaling not ridiculous
    std::pow(std::max(1.0,std::log(seeds.size())),2.0); 
    growPFClusters(seeds_to_topos,topoclusters,tolScal,0,tolScal,output);
  */
}
/*
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
    if( _convergencePosCalc ) {
      _convergencePosCalc->calculateAndSetPosition(current);
    } else {
      _positionCalc->calculateAndSetPosition(current);
    }
  }
}
*/

void ArborOnSeedsTopoClusterizer::
arborizeSeeds(const reco::PFRecHitCollection& rechits,
	      const std::vector<std::pair<unsigned,double> >& seeds,
	      arbor::branchcoll& branches) const {
  // arborize the seeds ~~~yeehaw~~~
  std::vector<TVector3> arborSeedHits;  
  for( const auto& seed : seeds ) {   
    const reco::PFRecHit& rh = rechits.at(seed.first);
    const auto& pos = rh.position();
    arborSeedHits.emplace_back(pos.x(),pos.y(),rh.depth());
  }
  branches = arbor::Arbor(arborSeedHits,_showerSigma,1);
  for( auto& branch : branches ) {
    std::sort(branch.begin(),branch.end(),
	      [&](const int a, const int b) {
		const reco::PFRecHit& ahit = rechits.at(seeds[a].first);
		const reco::PFRecHit& bhit = rechits.at(seeds[b].first);
		return ahit.position().mag2() < bhit.position().mag2();
	      });
  } // now all hits are ordered innermost to outermost  
}

void ArborOnSeedsTopoClusterizer::
getLinkedTopoClusters(const std::unordered_multimap<unsigned,unsigned>& topo_branch,
		      const std::unordered_multimap<unsigned,unsigned>& branch_topo,
		      const reco::PFClusterCollection& topoclusters,
		      const unsigned idx,
		      std::vector<bool>& used_topo,
		      std::vector<unsigned>& connected) const {
  if( used_topo[idx] ) return;
  connected.push_back(idx);
  used_topo[idx] = false;
  auto brange = topo_branch.equal_range(idx);
  for( auto bitr = brange.first; bitr != brange.second; ++bitr ) {
    auto trange = branch_topo.equal_range(idx);
    for( auto titr = trange.first; titr != trange.second; ++bitr ) {
      getLinkedTopoClusters(topo_branch,branch_topo,
			    topoclusters,
			    titr->second,
			    used_topo,
			    connected);
    }
  }
}

void ArborOnSeedsTopoClusterizer::
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
/*
void ArborOnSeedsTopoClusterizer::
growPFClusters(const std::unordered_multimap<unsigned,unsigned>&,
	       const reco::PFClusterCollection& topoclusters,
	       const unsigned toleranceScaling,
	       double diff,
	       reco::PFClusterCollection& clusters) const {     
  unsigned iter = 0;
  while( diff > _stoppingTolerance*toleranceScaling &&
	 iter <= _maxIterations ) {
    // reset the rechits in this cluster, keeping the previous position    
    std::vector<reco::PFCluster::REPPoint> clus_prev_pos;  
    for( auto& cluster : clusters) {
      const reco::PFCluster::REPPoint& repp = cluster.positionREP();
      clus_prev_pos.emplace_back(repp.rho(),repp.eta(),repp.phi());
      if( _convergencePosCalc ) {
	if( clusters.size() == 1 && _allCellsPosCalc ) {
	  _allCellsPosCalc->calculateAndSetPosition(cluster);
	} else {
	  _positionCalc->calculateAndSetPosition(cluster);
	}
      }
      cluster.resetHitsAndFractions();
    }
    // loop over topo cluster and grow current PFCluster hypothesis 
    std::vector<double> dist2, frac;
    double fractot = 0, fraction = 0;
    for( const reco::PFRecHitFraction& rhf : topo.recHitFractions() ) {
      const reco::PFRecHitRef& refhit = rhf.recHitRef();
      int cell_layer = (int)refhit->layer();
      if( cell_layer == PFLayer::HCAL_BARREL2 && 
	  std::abs(refhit->positionREP().eta()) > 0.34 ) {
	cell_layer *= 100;
      }  
      const double recHitEnergyNorm = 
	_recHitEnergyNorms.find(cell_layer)->second; 
      const math::XYZPoint& topocellpos_xyz = refhit->position();
      dist2.clear(); frac.clear(); fractot = 0;
      // add rechits to clusters, calculating fraction based on distance
      for( auto& cluster : clusters ) {      
	const math::XYZPoint& clusterpos_xyz = cluster.position();
	fraction = 0.0;
	const math::XYZVector deltav = clusterpos_xyz - topocellpos_xyz;
	const double d2 = deltav.Mag2()/_showerSigma2;
	dist2.emplace_back( d2 );
	if( d2 > 100 ) {
	  LOGDRESSED("Basic2DGenericPFlowClusterizer:growAndStabilizePFClusters")
	    << "Warning! :: pfcluster-topocell distance is too large! d= "
	    << d2;
	}
	// fraction assignment logic
	if( refhit->detId() == cluster.seed() && _excludeOtherSeeds ) {
	  fraction = 1.0;	
	} else if ( seedable[refhit.key()] && _excludeOtherSeeds ) {
	  fraction = 0.0;
	} else {
	  fraction = cluster.energy()/recHitEnergyNorm * vdt::fast_expf( -0.5*d2 );
	}      
	fractot += fraction;
	frac.emplace_back(fraction);
      }
      for( unsigned i = 0; i < clusters.size(); ++i ) {      
	if( fractot > _minFracTot || 
	    ( refhit->detId() == clusters[i].seed() && fractot > 0.0 ) ) {
	  frac[i]/=fractot;
	} else {
	  continue;
	}
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
	if( dist2[i] < 100.0 || frac[i] > 0.9999 ) {	
	  clusters[i].addRecHitFraction(reco::PFRecHitFraction(refhit,frac[i]));
	}
      }
    }
    // recalculate positions and calculate convergence parameter
    double diff2 = 0.0;  
    for( unsigned i = 0; i < clusters.size(); ++i ) {
      if( _convergencePosCalc ) {
	_convergencePosCalc->calculateAndSetPosition(clusters[i]);
      } else {
	if( clusters.size() == 1 && _allCellsPosCalc ) {
	  _allCellsPosCalc->calculateAndSetPosition(clusters[i]);
	} else {
	  _positionCalc->calculateAndSetPosition(clusters[i]);
	}
      }
      const double delta2 = 
	reco::deltaR2(clusters[i].positionREP(),clus_prev_pos[i]);    
      if( delta2 > diff2 ) diff2 = delta2;
    }
    diff = std::sqrt(diff2);
    dist2.clear(); frac.clear(); clus_prev_pos.clear();// avoid badness
    ++iter;
  }
}
*/
