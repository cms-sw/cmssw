#include "SharedSeedsClusterizer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <unordered_map>
#include <algorithm>
#include <iterator>

#include "vdt/vdtMath.h"

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

SharedSeedsClusterizer::
SharedSeedsClusterizer(const edm::ParameterSet& conf,
		       edm::ConsumesCollector& sumes) :
  InitialClusteringStepBase(conf,sumes),
  _useCornerCells(conf.getParameter<bool>("useCornerCells")),
  _showerSigma2(std::pow(conf.getParameter<double>("showerSigma"),2.0)),
  _stoppingTolerance(conf.getParameter<double>("stoppingTolerance")),
  _minFracTot(conf.getParameter<double>("minFracTot")),
  _maxIterations(conf.getParameter<unsigned>("maxIterations")) { 
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

void SharedSeedsClusterizer::
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
    if( temp.recHitFractions().size() ) {
      //std::cout << "topo cluster: " << temp  << std::endl;
      topoclusters.push_back(temp);
    }
  }  
  
  // cells with no neighbours in the previous layer are non-shared 
  // "primary" seeds, all others are secondary that get shared 
  // by the log-weighted fraction of seed energies
  // create a new PFcluster for 
  seed_usage_map seed_has_weights;
  seed_fractions_map seeds_and_fractions;
  for( unsigned i = 0; i < seeds.size(); ++i ) {        
    seed_has_weights.emplace(seeds[i].first,false);
  }  
  for( const std::vector<unsigned>& branch : branches ) {
    for( auto ihit = branch.rbegin(); ihit != branch.rend(); ++ihit) {  
      if( !seed_has_weights[*ihit] ) {  
	buildInitialWeightsList(inp,seedable,seedtypes,
				branches,*ihit,
				seed_has_weights,
				seeds_and_fractions);	
      }
    }
  }
	
  // flatten the maps into vector of branches with fractions
  // create map of seeds -> pfclusters
  unsigned ibranch = 0;
  std::unordered_multimap<unsigned,unsigned> seeds_to_clusters;
  std::vector<std::vector<std::pair<unsigned,double> > > seeds_with_weights;
  for( const auto& seed : seeds ) {    
    if( seedtypes[seed.first] == PrimarySeed ) {
      std::vector<std::pair<unsigned,double> > abranch;
      abranch.emplace_back(seed.first,
			   seeds_and_fractions[seed.first][seed.first]);
      auto range = seeds_to_branches.equal_range(seed.first);
      if( std::distance(range.first,range.second) != 1 ) {
	throw cms::Exception("ImpossibleOutcome") 
	  << "seed in more than one connected set!" << std::endl;
      }
      const auto& connected = branches[range.first->second];
      for( const unsigned maybe : connected ) {
	
	if( maybe == seed.first ) continue; // already added primary seed!
	auto& fracs = seeds_and_fractions[maybe];
	if( fracs.find(seed.first) != fracs.end() ) {
	//std::cout << maybe  
	//	    << " connected to " << abranch[0].first
	//	    << " with fraction " << fracs[seed.first] << std::endl;
	seeds_to_clusters.emplace(maybe,ibranch);
	  abranch.emplace_back(maybe,fracs[seed.first]);
	}
      }
      std::sort(abranch.begin()+1,abranch.end(),
		[&](const std::pair<unsigned,double>& a,
		    const std::pair<unsigned,double>& b) {
		  return inp[a.first].depth() < inp[b.first].depth();
		});
      seeds_with_weights.push_back(std::move(abranch));      
      ++ibranch;
    }    
  } 
      
  // now we turn the branches with weights into PFClusters
  for( const auto& branch : seeds_with_weights ) {
    // the first hit in the branch is the primary seed which always
    // has weight one
    output.push_back(reco::PFCluster());
    reco::PFCluster& current = output.back();    
    reco::PFRecHitFraction rhf(makeRefhit(input,branch[0].first),1.0);
    double seed_energy = rhf.recHitRef()->energy();
    current.addRecHitFraction(rhf);
    current.setSeed(rhf.recHitRef()->detId());
    // all other hits in the branch are secondary and have their fraction
    // determined by log-weights (but these fractions do not evolve in the 
    // position fit)
    for( auto rhit = branch.begin()+1; rhit != branch.end(); ++rhit ) {
      reco::PFRecHitFraction rhf2(makeRefhit(input,rhit->first),rhit->second);
      current.addRecHitFraction(rhf2);   
      const double rh_energy = rhf2.recHitRef()->energy()*rhf2.fraction();
      if( rh_energy  > seed_energy ) {
	seed_energy = rh_energy;
	current.setSeed(rhf2.recHitRef()->detId());
      }
    }    
    _positionCalc->calculateAndSetPosition(current);
    //std::cout << "Seeded cluster : " << current << std::endl;
  }  
  
  // connect together per-layer topo clusters with the new pf clusters
  std::unordered_multimap<unsigned,unsigned> clusters_to_topos;
  std::unordered_multimap<unsigned,unsigned> topos_to_clusters;
  for( unsigned i = 0 ; i < topoclusters.size(); ++i  ) {
    for( const auto& rhf : topoclusters[i].recHitFractions() ) {
      if( !seedable[rhf.recHitRef().key()] ) continue;
      for( unsigned j = 0; j < output.size(); ++j ) {
	auto ihit = std::find_if(output[j].recHitFractions().begin(),
				 output[j].recHitFractions().end(),
				 [&](const reco::PFRecHitFraction& hit){
				   if( hit.recHitRef().key() == 
				       rhf.recHitRef().key() ) return true;
				   return false;
				 });
	if( ihit != output[j].recHitFractions().end() ) {
	  topos_to_clusters.emplace(i,j);	  
	  clusters_to_topos.emplace(j,i);
	}
      }
    }
  }
  // have relational maps of topo clusters grouped together and 
  // constituent branches (shitty code and probably very slow right now)
  std::vector<std::vector<unsigned> > grouped_topos; // same indices
  std::vector<std::vector<unsigned> > grouped_clusters; // same indices
  std::vector<bool> used_topo(topoclusters.size(),false);
  for(unsigned i = 0; i < topoclusters.size(); ++i ) {
    if( used_topo[i] ) continue; //skip topo clusters already used
    grouped_topos.push_back(std::vector<unsigned>());
    getLinkedTopoClusters( topos_to_clusters,
			   clusters_to_topos,
			   topoclusters,
			   i,
			   used_topo,
			   grouped_topos.back() );
  }
  for( unsigned i = 0 ; i < grouped_topos.size(); ++i ) {
    grouped_clusters.push_back(std::vector<unsigned>());
    std::vector<unsigned>& current = grouped_clusters.back();
    for( unsigned itopo : grouped_topos[i] ) {
      auto cluster_range = topos_to_clusters.equal_range(itopo);
      for( auto ic = cluster_range.first; ic != cluster_range.second; ++ic ) {
	auto branchid = std::find(current.begin(),current.end(),ic->second);	
	if( branchid == current.end() ) {
	  current.push_back(ic->second);
	}
      }
    }
  }  
  
  // run semi-3D (2D position fit with center-per-layer given by average) 
  // pf cluster position fit, loop on seeds
  const unsigned tolScal = //take log to make scaling not ridiculous
    std::pow(std::max(1.0,std::log(seeds.size()+1)),2.0); 
  growPFClusters(grouped_clusters,grouped_topos,topos_to_clusters,
		 seedable,topoclusters,tolScal,tolScal,output);
  
  /*
  std::cout << "made " << output.size() << " clusters!" << std::endl;
  for( const auto& cluster : output ) {
    std::cout << "Final cluster: " << cluster << std::endl;
  }
  */
  
  

}

void SharedSeedsClusterizer::
linkSeeds(const reco::PFRecHitCollection& rechits,
	  const std::vector<bool>& seedable,
	  const std::vector<std::pair<unsigned,double> >& seeds,
	  std::unordered_multimap<unsigned,unsigned>& seeds_to_branches,
	  std::vector<seed_type>& seed_types, // assumed to be all "not a seed"
	  std::vector<std::vector<unsigned> >& branches) const {
  std::vector<bool> used_seed(rechits.size(),false);
  std::vector<std::vector<unsigned> > linked_seeds; 
  unsigned ibranch = 0;
  for( const auto& seed : seeds ) {
    if( !used_seed[seed.first] ) {
      std::vector<unsigned> current ;
      findSeedNeighbours(rechits,seedable,ibranch,seed.first,used_seed,
			 seeds_to_branches,current);
      std::sort( current.begin(), current.end(), 
		 [&](unsigned a, unsigned b){
		   return rechits[a].depth() < rechits[b].depth();
		 });
      branches.push_back(std::move(current));
      ++ibranch;
    }
  }
  
  for( const auto& branch : branches ) {
    for( const unsigned hit : branch ) {
      
      bool is_primary = true;
      const reco::PFRecHitRefVector& neighbs = rechits[hit].neighbours();
      const std::vector<unsigned short>& nb_info = rechits[hit].neighbourInfos();
      for( unsigned i = 0; i < neighbs.size(); ++i ) {
	if( !seedable[neighbs[i].key()] ) continue;
	if( ( (nb_info[i] >> 9) & 0x3 ) == 1 &&
	    ( (nb_info[i] >> 8) & 0x1 ) == 0 ) is_primary = false;
      }
      seed_types[hit] = is_primary ? PrimarySeed : SecondarySeed; 
      
      /*
      std::cout << hit << " is a " << ( is_primary ? "primary" : "secondary" )
		<< " seed! " << rechits[hit].energy() 
		<< ' ' << rechits[hit].positionREP() << std::endl;
      */
      
    }

    /*
    std::cout << "--- initial seeds list ---" << std::endl;
    for( unsigned seed : branch ) {
      const reco::PFRecHit& hit = rechits[seed];
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
    */
  }
}
// find all depth-wise neighbours for this seed
void SharedSeedsClusterizer::
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

void SharedSeedsClusterizer::
buildInitialWeightsList(const reco::PFRecHitCollection& rechits,
			const std::vector<bool>& seedable,
			const std::vector<seed_type>& seedtypes,
			const std::vector<std::vector<unsigned> >& linked_seeds,
			const unsigned  seed_idx,
			seed_usage_map& has_weight_data,
			seed_fractions_map& resolved_seeds) const {
  // handle call termination
  switch( seedtypes[seed_idx] ) {
  case PrimarySeed:
    if( !has_weight_data[seed_idx] ) {
      has_weight_data[seed_idx] = true;
      resolved_seeds[seed_idx].emplace(seed_idx,1.0);
    }
    return;
    break;
  case SecondarySeed:
    if( has_weight_data[seed_idx] ) return;
    break;
  default:
    throw cms::Exception("UnexpectedHit") << "Non-seed hit encountered!";
  }
  // get all back-links of the present hit
  std::vector<bool> temp(rechits.size(),false);
  std::unordered_multimap<unsigned,unsigned> temp2;
  std::vector<unsigned> backlinks;
  std::vector<unsigned> myprimaries;
  findSeedNeighbours(rechits,seedable,0,seed_idx,temp,
		     temp2,backlinks,OnlyBackward);
  // if something doesn't have weight data, calculate it recursively
  for( auto ihit = backlinks.begin()+1; ihit != backlinks.end(); ++ihit ) {
    if( !has_weight_data[*ihit] ) buildInitialWeightsList(rechits,
							  seedable,
							  seedtypes,
							  linked_seeds,
							  *ihit,
							  has_weight_data,
							  resolved_seeds);
    if( seedtypes[*ihit] == PrimarySeed ) myprimaries.push_back(*ihit);
  }
  std::unordered_map<unsigned,double> branch_energies;
  for( const unsigned primary : myprimaries ) {
     // std::cout << "building energies for primary seed: " 
	//      << primary << std::endl;
    
    branch_energies[primary] = 0.0;
    std::vector<bool> temp3(rechits.size(),false);
    std::unordered_multimap<unsigned,unsigned> temp4;
    std::vector<unsigned> forwardlinks;
    std::vector<unsigned> path;
    findSeedNeighbours(rechits,seedable,0,primary,temp3,
		       temp4,forwardlinks,OnlyForward);
    // get the seeds in the path from this primary seed to whatever we're 
    // trying to determine weights for at the moment
    for( const unsigned forwardlink : forwardlinks ) {
      for( const unsigned backlink : backlinks ) {
	if( forwardlink == backlink ) path.push_back(forwardlink);
      }
    }
    for( const unsigned previous : path ) {
      if( previous == seed_idx || 
	  ( seedtypes[previous] == PrimarySeed && previous != primary ) ) {
	continue;
      }
      const double fraction = resolved_seeds[previous][primary];
      const reco::PFRecHit& rh = rechits[previous];      
      // minimum energy in fraction of 0.05 GeV
      const double rhenergy = rh.energy()*fraction;
      //std::cout << primary << ' ' << previous << ' ' << seed_idx << ' '
      //	<< rh.energy() << ' ' << rhenergy << ' ' << fraction 
      //	<< std::endl;
      branch_energies[primary] += rhenergy;
    }
  }
  double log_energy_sum = 0.0;
  for( const auto& branch_and_E : branch_energies ) {
    log_energy_sum += std::log(20.0*branch_and_E.second);
  }
  for( const auto& branch_and_E : branch_energies ) {
    resolved_seeds[seed_idx][branch_and_E.first] = 
      std::log(20.0*branch_and_E.second)/log_energy_sum; 
  }
  has_weight_data[seed_idx] = true;
}


void SharedSeedsClusterizer::
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

void SharedSeedsClusterizer::
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

// given seeded pfclusters that are topologically associated to each other 
// (last argument) loop through each layer and determine seed weights
// -- secondary seeds are not shared by distance
// modification for later -- use seeds per-cluster per-layer to create
// gaussian centroid pulled away from the shower axis
// ----- grouped_topos and grouped_clusters *must* have the same indices
void SharedSeedsClusterizer::
growPFClusters(const std::vector<std::vector<unsigned> >& grouped_clusters,
	       const std::vector<std::vector<unsigned> >& grouped_topos,
	       const std::unordered_multimap<unsigned,unsigned>& topos_to_clusters,
	       const std::vector<bool>& seedable,
	       const reco::PFClusterCollection& topoclusters,
	       const unsigned toleranceScaling,
	       double diff,
	       reco::PFClusterCollection& clusters) const {     
  const double start_diff = diff;
  unsigned igroup = 0;
  for( const auto& cluster_group : grouped_clusters ) {
    unsigned iter = 0;
    diff = start_diff;
    while( diff > _stoppingTolerance*toleranceScaling &&
	   iter <= _maxIterations ) {
      // reset the rechits in this cluster, keeping the previous position    
      std::vector<reco::PFCluster::REPPoint> clus_prev_pos;  
      for( const unsigned cluster_idx : cluster_group) {
	reco::PFCluster& cluster = clusters[cluster_idx];
	const reco::PFCluster::REPPoint& repp = cluster.positionREP();
	clus_prev_pos.emplace_back(repp.rho(),repp.eta(),repp.phi());
	if( cluster_group.size() == 1 && _allCellsPosCalc ) {
	  _allCellsPosCalc->calculateAndSetPosition(cluster);
	} else {
	  _positionCalc->calculateAndSetPosition(cluster);
	}
	// reset the cluster except for seeds, skip seeds in pos calc loop
	cluster.pruneUsing([&](const reco::PFRecHitFraction& rhf){
	    return seedable[rhf.recHitRef().key()];
	  });
      }
      // loop over all grouped topo clusters and update each individually
      for( const unsigned topo_idx : grouped_topos[igroup] ) {
	const reco::PFCluster& topo = topoclusters[topo_idx];
	// loop over this topo cluster and grow current PFCluster hypothesis
	std::vector<double> dist2, frac;
	double fractot = 0, fraction = 0;
	for( const reco::PFRecHitFraction& rhf : topo.recHitFractions() ) {
	  const reco::PFRecHitRef& refhit = rhf.recHitRef();
	  // skip all seeds, their fractions are fixed
	  if( seedable[refhit.key()] ) continue; 
	  // used standard distance based weighting in each layer
	  int cell_layer = (int)refhit->layer();
	  if( cell_layer == PFLayer::HCAL_BARREL2 && 
	      std::abs(refhit->positionREP().eta()) > 0.34 ) {
	    cell_layer *= 100;
	  }  
	  const double recHitEnergyNorm = 0.05;
	    //_thresholds.find(cell_layer)->second.first; 
	  const math::XYZPoint& topocellpos_xyz = refhit->position();
	  dist2.clear(); frac.clear(); fractot = 0;
	  // add rechits to clusters, calculating fraction based on distance
	  auto topo_r = topos_to_clusters.equal_range(topo_idx);
	  for( auto clstr = topo_r.first; clstr != topo_r.second; ++clstr ) {
	    reco::PFCluster& cluster = clusters[clstr->second];
	    //need to take the cluster rho/eta/phi and 
	    //adjust to this layer's rho
	    reco::PFCluster::REPPoint clusterpos_rep = cluster.positionREP();
	    clusterpos_rep.SetRho(topocellpos_xyz.Rho());
	    const math::XYZPoint clusterpos_xyz(clusterpos_rep);
	    fraction = 0.0;
	    const math::XYZVector deltav = clusterpos_xyz - topocellpos_xyz;
	    const double d2 = deltav.Mag2()/_showerSigma2;
	    dist2.emplace_back( d2 );
	    if( d2 > 100 ) {
	      LOGDRESSED("Basic2DGenericPFlowClusterizer:growAndStabilizePFClusters")
		<< "Warning! :: pfcluster-topocell distance is too large! d= "
		<< d2;
	    }
	    // fraction assignment logic (remember we are skipping all seeds)
	    fraction = cluster.energy()/recHitEnergyNorm * vdt::fast_expf( -0.5*d2 );
	    fractot += fraction;
	    frac.emplace_back(fraction);
	  }
	  for( auto clstr = topo_r.first; clstr != topo_r.second; ++clstr ) {
	    unsigned i = std::distance(topo_r.first,clstr);	  
	    if( fractot > _minFracTot ) {
	      double temp = frac[i];
	      frac[i]/=fractot;
	      if( frac[i] > 1.0 )  {
		throw cms::Exception("InvalidFraction") 
		  << "Fraction is larger than 1!!! "
		  << " Ingredients : " << temp << ' ' 
		  <<  fractot << ' ' << std::sqrt(dist2[i]) << ' ' 
		  << clusters[clstr->second].energy() << std::endl;
	      }
	    }
	    else continue;
	    if( dist2[i] < 100.0 || frac[i] > 0.9999 ) {	
	      clusters[clstr->second].addRecHitFraction(reco::PFRecHitFraction(refhit,frac[i]));
	    }
	  }
	} // loop on topo cluster rechit fractions
	dist2.clear(); frac.clear(); //clus_prev_pos.clear();
      } // end of loop on associated topological clusters
      // recalculate positions and calculate convergence parameter
      double diff2 = 0.0;  
      for( unsigned i = 0; i < cluster_group.size(); ++i ) {
	reco::PFCluster& cluster = clusters[cluster_group[i]];
	if( cluster_group.size() == 1 && _allCellsPosCalc ) {
	  _allCellsPosCalc->calculateAndSetPosition(cluster);
	} else {
	  _positionCalc->calculateAndSetPosition(cluster);
	}
	const double delta2 = 
	  reco::deltaR2(cluster.positionREP(),clus_prev_pos[i]);    
	if( delta2 > diff2 ) diff2 = delta2;
      }
      diff = std::sqrt(diff2);         
      ++iter;
    }
    ++igroup;
  }
}

