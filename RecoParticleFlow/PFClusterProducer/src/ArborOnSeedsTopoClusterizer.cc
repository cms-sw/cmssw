#include "ArborOnSeedsTopoClusterizer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <unordered_map>
#include <algorithm>
#include <iterator>

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

ArborOnSeedsTopoClusterizer::
ArborOnSeedsTopoClusterizer(const edm::ParameterSet& conf) :
  InitialClusteringStepBase(conf),
  _useCornerCells(conf.getParameter<bool>("useCornerCells")),
  _showerSigma(conf.getParameter<double>("showerSigma")) { 
  _positionCalc.reset(NULL);
  if( conf.exists("positionCalc") ) {
    const edm::ParameterSet& pcConf = conf.getParameterSet("positionCalc");
    const std::string& algo = pcConf.getParameter<std::string>("algoName");
    PosCalc* calcp = PFCPositionCalculatorFactory::get()->create(algo, 
								 pcConf);
    _positionCalc.reset(calcp);
  }
  _allCellsPosCalc.reset(NULL);
  if( conf.exists("allCellsPositionCalc") ) {
    const edm::ParameterSet& acConf = 
      conf.getParameterSet("allCellsPositionCalc");
    const std::string& algoac = 
      acConf.getParameter<std::string>("algoName");
    PosCalc* accalc = 
      PFCPositionCalculatorFactory::get()->create(algoac, acConf);
    _allCellsPosCalc.reset(accalc);
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
  std::vector<std::vector<unsigned> > branches; 
  std::vector<seed_type> seedtypes(inp.size(),NotSeed);
  std::unordered_multimap<unsigned,unsigned> seeds_to_branches;
  branches.reserve(seeds.size());
  linkSeeds(inp,seedable,seeds,seeds_to_branches,seedtypes,branches);

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
				 [&](const unsigned& hit){
				   if( hit == 
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
  std::vector<bool> used_topo(topoclusters.size(),false);
  std::cout << topoclusters.size() << ' ' 
	    << branches.size() << std::endl;
  for(unsigned i = 0; i < topoclusters.size(); ++i ) {
    if( used_topo[i] ) continue; //skip topo clusters already used
    grouped_topos.push_back(std::vector<unsigned>());
    getLinkedTopoClusters( topos_to_branches,
			   branches_to_topos,
			   topoclusters,
			   i,
			   used_topo,
			   grouped_topos.back() );
  }
  for( unsigned i = 0 ; i < grouped_topos.size(); ++i ) {
    grouped_branches.push_back(std::vector<unsigned>());
    std::vector<unsigned>& current = grouped_branches.back();
    for( unsigned itopo : grouped_topos[i] ) {
      auto branch_range = topos_to_branches.equal_range(itopo);
      for( auto ib = branch_range.first; ib != branch_range.second; ++ib ) {
	auto branchid = std::find(current.begin(),current.end(),ib->second);	
	if( branchid == current.end() ) {
	  current.push_back(ib->second);
	}
      }
    }
  }

  for( unsigned i = 0; i < grouped_topos.size(); ++i ) {
    std::cout << i << " grouped topos: ";
    for( unsigned topo : grouped_topos[i] ) std::cout << topo << ",";
    std::cout << std::endl;
    std::cout << i << " grouped branches: ";
    for( unsigned branch : grouped_branches[i] ) std::cout << branch << ",";
    std::cout << std::endl;
  }
  
  // cells with no neighbours in the previous layer are non-shared 
  // "primary" seeds, all others are secondary that get shared 
  // by the log-weighted fraction of seed energies
  // create a new PFcluster for 
  std::vector<std::vector<std::pair<unsigned,double> > > seeds_with_weights;
  calculateInitialWeights(inp,seedable,seedtypes,seeds_to_branches,
			  branches,seeds_with_weights);

  // now we turn the branches with weights into PFClusters
  for( const auto& branch : seeds_with_weights ) {
    // the first hit in the branch is the primary seed which always
    // has weight one
    std::cout << "-- new branch --" << std::endl;
    output.push_back(reco::PFCluster());
    reco::PFCluster& current = output.back();
    reco::PFRecHitFraction rhf(makeRefhit(input,branch[0].first),1.0);
    current.addRecHitFraction(rhf);
    current.setSeed(rhf.recHitRef()->detId());
    std::cout << rhf.recHitRef()->depth() << ' ' 
	      << rhf.recHitRef()->position().R() << ' '
	      << rhf.recHitRef()->layer() << ' ' 
	      << rhf.recHitRef()->positionREP() << std::endl;
    // all other hits in the branch are secondary and have their fraction
    // determined by log-weights (but these fractions do not evolve in the 
    // position fit)
    for( auto rhit = branch.begin()+1; rhit != branch.end(); ++rhit ) {
      reco::PFRecHitFraction rhf2(makeRefhit(input,rhit->first),rhit->second);
      current.addRecHitFraction(rhf2);
      std::cout << rhf2.recHitRef()->depth() << ' ' 
		<< rhf2.recHitRef()->position().R() << ' ' 
		<< rhf2.recHitRef()->layer() << ' '
		<< rhf2.recHitRef()->positionREP() << std::endl;
    }    
    _positionCalc->calculateAndSetPosition(current);
  }

  /*
  // run semi-3D (2D position fit with center-per-layer given by 3D line fit) 
  // pf cluster position fit, loop on seeds
  const unsigned tolScal = //take log to make scaling not ridiculous
    std::pow(std::max(1.0,std::log(seeds.size())),2.0); 
    growPFClusters(seeds_to_topos,topoclusters,tolScal,0,tolScal,output);
  */
}

void ArborOnSeedsTopoClusterizer::
positionCalc( const reco::PFClusterCollection& topo_clusters,
	      const std::vector<unsigned>& topo_indices,
	      const std::vector<unsigned>& branch_indices,
	      reco::PFClusterCollection& clusters ) const {
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
linkSeeds(const reco::PFRecHitCollection& rechits,
	  const std::vector<bool>& seedable,
	  const std::vector<std::pair<unsigned,double> >& seeds,
	  std::unordered_multimap<unsigned,unsigned>& seeds_to_branches,
	  std::vector<seed_type>& seed_types, // assumed to be all "not a seed"
	  std::vector<std::vector<unsigned> >& branches) const {
  std::vector<bool> used_seed(rechits.size(),false);
  std::vector<std::vector<unsigned> > linked_seeds;
  
  for( const auto& seed : seeds ) {
    if( !used_seed[seed.first] ) {
      linked_seeds.push_back(std::vector<unsigned>());
      std::unordered_multimap<unsigned,unsigned> t2;
      std::vector<unsigned>& current = linked_seeds.back();
      findSeedNeighbours(rechits,seedable,0,seed.first,used_seed,t2,current);
      std::sort( current.begin(), current.end(), 
		 [&](unsigned a, unsigned b){
		   return rechits[a].depth() < rechits[b].depth();
		 });
    }
  } 
  
  unsigned ibranch = 0;
  for( const auto& branch : linked_seeds ) {
    for( const unsigned hit : branch ) {
      bool is_primary = true;
      seed_types[hit] = SecondarySeed;
      const reco::PFRecHitRefVector& neighbs = rechits[hit].neighbours();
      const std::vector<unsigned short>& nb_info = rechits[hit].neighbourInfos();
      for( unsigned i = 0; i < neighbs.size(); ++i ) {
	if( !seedable[neighbs[i].key()] ) continue;
	if( ( (nb_info[i] >> 9) & 0x3 ) == 1 &&
	    ( (nb_info[i] >> 8) & 0x1 ) == 0 ) is_primary = false;
      }
      if( is_primary ) { 
	std::cout << hit << " is a primary seed!" << std::endl;
	seed_types[hit] = PrimarySeed;	
	std::vector<unsigned> temp;
	std::vector<bool> t3(rechits.size(),false);
	// get links to all *deeper* seeds (we must go deeper!)
	findSeedNeighbours(rechits,seedable,ibranch,hit,t3,
			   seeds_to_branches,temp,OnlyForward);	
	std::sort(temp.begin(),temp.end(),
		  [&](const unsigned a,
		      const unsigned b){
		    return rechits[a].depth() < rechits[b].depth();
		  });
	std::cout << "forward-linked hits: ";
	for( const unsigned secondary : temp ) {
	  std::cout << secondary << ",";
	}
	std::cout << std::endl;	
	branches.push_back(std::move(temp));
	++ibranch;
      }
    }

    

    std::cout << "--- initial seeds list ---" << std::endl;
    std::vector<unsigned> depths;
    for( unsigned seed : branch ) {
      const reco::PFRecHit& hit = rechits[seed];
      depths.push_back(hit.depth());
      std::cout << seed << ' ' << hit.positionREP() 
		<< ' ' << hit.position().R() << ' ' 
		<< hit.depth() << ' ' << hit.energy() << std::endl;
      const reco::PFRecHitRefVector& neighbs = hit.neighbours();
      const std::vector<unsigned short>& nb_info = hit.neighbourInfos();
      for( unsigned i = 0; i < neighbs.size(); ++i ) {
	if( !seedable[neighbs[i].key()] ) continue;
	int z = (nb_info[i] >> 9) & 0x3;
	if( ( (nb_info[i] >> 8) & 0x1 ) == 0 ) z = -z;
	switch(z) {
	case 1: 
	  std::cout << '\t' << neighbs[i].key() 
		    << " is a neighbour in the next layer" << std::endl;
	  break;
	case 0:
	  std::cout << '\t' << neighbs[i].key() 
		    << " is a neighbour in this layer" << std::endl;
	  break;
	case -1:
	  std::cout << '\t' << neighbs[i].key() 
		    << " is a neighbour in the previous layer" << std::endl;
	  break;
	default :
	  std::cout << "weird!" << std::endl;
	}
      }
    }
  }

}
// find all depth-wise neighbours for this seed
void ArborOnSeedsTopoClusterizer::
findSeedNeighbours(const reco::PFRecHitCollection& hits,
		   const std::vector<bool>& seedable,
		   const unsigned branch_idx,
		   const unsigned seed_idx,
		   std::vector<bool>& used_seed,
		   std::unordered_multimap<unsigned,unsigned>& seeds_to_branches,
		   std::vector<unsigned>& connected_seeds,
		   navi_dir direction) const{
  if( used_seed[seed_idx] ) return;
  connected_seeds.push_back(seed_idx);
  seeds_to_branches.emplace(seed_idx,branch_idx);
  used_seed[seed_idx] = true;
  
  const reco::PFRecHitRefVector& neighbs = hits[seed_idx].neighbours();
  const std::vector<unsigned short>& nb_info = hits[seed_idx].neighbourInfos();
  for( unsigned i = 0; i < neighbs.size(); ++i ) {
    unsigned absz = (nb_info[i] >> 9) & 0x3;
    unsigned sgnz = (nb_info[i] >> 8) & 0x1;
    //unsigned absx = (nb_info[i] >> 1) & 0x3;
    //unsigned absy = (nb_info[i] >> 5) & 0x3;
    unsigned nb_idx = neighbs[i].key();
    // skip non-seed neighbours or those on same layer
    if( seedable[nb_idx] && absz == 1 && 
	( direction != OnlyForward ||  sgnz != 0 ) &&
	( direction != OnlyBackward || sgnz == 0 ) ) {
      findSeedNeighbours(hits,seedable,branch_idx,nb_idx,used_seed,
			 seeds_to_branches,connected_seeds,direction);
    }
  }
  
}

void ArborOnSeedsTopoClusterizer::
calculateInitialWeights(const reco::PFRecHitCollection& rechits,
			const std::vector<bool>& seedable,
			const std::vector<seed_type>& seedtypes,
			const std::unordered_multimap<unsigned,unsigned>& seeds_to_branches,
			const std::vector<std::vector<unsigned> >& raw_branches,
			std::vector<std::vector<std::pair<unsigned,double> > >& weighted_branches) const {
  std::unordered_map<unsigned,std::pair<unsigned,double> > fractions_in_branches;
  for( unsigned i = 0; i < raw_branches.size(); ++i ) {
    const std::vector<unsigned>& branch = raw_branches[i];
    weighted_branches.push_back(std::vector<std::pair<unsigned,double> >());
    std::vector<std::pair<unsigned,double> >& br = weighted_branches.back();
    // first hit in branch is always the primary seed of the branch
    br.emplace_back(branch[0],1.0);
    // all other hits are secondary seeds and can be shared
    // first we must determine what primary seeds are linked 
    // to this secondary seed
    // after that we need to walk from each primary seed to this secondary
    // seed to determine the energy of each cluster that is infront of
    // this seed, the energy of the secondary seed is then split by 
    // a log-energy-weight average of each contributing cluster's energy
    for( auto ihit = branch.begin()+1; ihit != branch.end(); ++ihit ) {
      double log_E_sum_this_seed = 0.0;
      double log_E_sum_all_seeds = 0.0;
      // to get the connection between a seed and this hit just ask for
      // the intersection of the branch and the backward-facing hit
      // list from this secondary seed
      // first do the branch we are presently working on     
      auto branches = seeds_to_branches.equal_range(*ihit);
      for( auto obr = branches.first; obr != branches.second; ++obr )  {
	std::vector<unsigned> path;
	std::vector<unsigned> backlinks;
	std::unordered_multimap<unsigned,unsigned> t2;
	std::vector<bool> t3(rechits.size(),false);
	double log_E_sum = 0.0;
	findSeedNeighbours(rechits,seedable,0,*ihit,t3,
			   t2,backlinks,OnlyBackward);		
	std::cout << "found hits in path between "
		  << raw_branches[obr->second][0] << " and " << *ihit 
		  << " : " << std::endl;
	for( const unsigned hit_in_path : path ) {
	  if( hit_in_path == branch[0] ) {
	    log_E_sum += std::log(2.0*rechits[hit_in_path].energy());	    
	  } else if ( seedtypes[hit_in_path] == SecondarySeed ) {
	    //const std::pair<unsigned,double>& hit_and_fraction;	    
	    log_E_sum += std::log(2.0);
	  }
	}
	std::cout << std::endl;
	if( obr->second == i ) log_E_sum_this_seed = log_E_sum;
	log_E_sum_all_seeds += log_E_sum;
      }
      br.emplace_back(*ihit,log_E_sum_this_seed/log_E_sum_all_seeds);
    }
  }  
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
  used_topo[idx] = true;
  auto brange = topo_branch.equal_range(idx);
  for( auto bitr = brange.first; bitr != brange.second; ++bitr ) {
    auto trange = branch_topo.equal_range(bitr->second);
    for( auto titr = trange.first; titr != trange.second; ++titr ) {
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
