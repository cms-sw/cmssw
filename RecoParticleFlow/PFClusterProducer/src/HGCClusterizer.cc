#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

// helpful tools
#include "KDTreeLinkerAlgoT.h"
#include <unordered_map>
#include <unordered_set>

// point here is that we find EM-like clusters first and build those
// then remove rechits from the pool and find the Had-like 
// clusters in some way

class HGCClusterizer : public InitialClusteringStepBase {
  typedef HGCClusterizer B2DGT;
  typedef KDTreeLinkerAlgo<unsigned,3> KDTree;
  typedef KDTreeNodeInfoT<unsigned,3> KDNode;
  typedef PFCPositionCalculatorBase PosCalc;
  typedef std::pair<unsigned,unsigned> HitLink;
  typedef std::unordered_multimap<unsigned,unsigned> LinkMap;
  typedef std::unordered_set<unsigned> UniqueIndices;
 public:
  HGCClusterizer(const edm::ParameterSet& conf);
  virtual ~HGCClusterizer() {}
  HGCClusterizer(const B2DGT&) = delete;
  B2DGT& operator=(const B2DGT&) = delete;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
		     const std::vector<bool>&,
		     const std::vector<bool>&, 
		     reco::PFClusterCollection&);
  
private:
  // used for rechit searching 
  std::vector<KDNode> _nodes, _found;
  KDTree _kdtree;

  std::unique_ptr<PosCalc> _logWeightPosCalc,_pcaPosCalc;

  std::array<float,3> _moliere_radii;

  // helper functions for various steps in the clustering
  void build2DCluster(const reco::PFRecHitCollection&,
		      const std::vector<bool>&,
		      const std::vector<bool>&,		     
		      const reco::PFRecHitRef&,
		      std::vector<bool>&,
		      reco::PFCluster&);  

  // linking in Z for clusters
  void linkClustersInLayer(const reco::PFClusterCollection& input_clusters,
		      reco::PFClusterCollection& output);

  // utility
  reco::PFRecHitRef makeRefhit( const edm::Handle<reco::PFRecHitCollection>& h,
                                const unsigned i ) const {
    return reco::PFRecHitRef(h,i);
  }
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory,
		  HGCClusterizer,
		  "HGCClusterizer");

namespace {
  std::pair<float,float> minmax(const float a, const float b) {
    return ( b < a ? 
	     std::pair<float,float>(b, a) : 
	     std::pair<float,float>(a, b)   );
  }

  bool greaterByEnergy(const std::pair<unsigned,double>& a,
		       const std::pair<unsigned,double>& b) {
    return a.second > b.second;
  }

  class QuickUnion{    
  public:
    QuickUnion(const unsigned NBranches) {
      _count = NBranches;
      _id.resize(NBranches);
      _size.resize(NBranches);
      for( unsigned i = 0; i < NBranches; ++i ) {
	_id[i] = i;
	_size[i] = 1;
      }
    }
    
    int count() const { return _count; }
    
    unsigned find(unsigned p) {
      while( p != _id[p] ) {
	_id[p] = _id[_id[p]];
	p = _id[p];
      }
      return p;
    }
    
    bool connected(unsigned p, unsigned q) { return find(p) == find(q); }
    
    void unite(unsigned p, unsigned q) {
      unsigned rootP = find(p);
      unsigned rootQ = find(q);
      _id[p] = q;
      
      if(_size[rootP] < _size[rootQ] ) { 
	_id[rootP] = rootQ; _size[rootQ] += _size[rootP]; 
      } else { 
	_id[rootQ] = rootP; _size[rootP] += _size[rootQ]; 
      }
      --_count;
    }
    std::vector<unsigned> _id;
    std::vector<unsigned> _size;
    int _count;
    
  };
}

HGCClusterizer::HGCClusterizer(const edm::ParameterSet& conf) :
    InitialClusteringStepBase(conf) { 
  // clean initial state for searching
  _nodes.clear(); _found.clear(); _kdtree.clear();
  // setup 2D position calculator for per-layer clusters
  _logWeightPosCalc.reset(nullptr); 
  const edm::ParameterSet& pconf = conf.getParameterSet("positionCalcInLayer");
  const std::string& algo = pconf.getParameter<std::string>("algoName");
  PosCalc* calc = PFCPositionCalculatorFactory::get()->create(algo,pconf);
  _logWeightPosCalc.reset(calc);
  // setup PCA position calculator for full cluster
  _pcaPosCalc.reset(nullptr); 
  const edm::ParameterSet& pcaconf = conf.getParameterSet("positionCalcPCA");
  const std::string& pcaalgo = pcaconf.getParameter<std::string>("algoName");
  PosCalc* pcacalc = 
    PFCPositionCalculatorFactory::get()->create(pcaalgo,pcaconf);
  _pcaPosCalc.reset(pcacalc);
  // get moliere radius and nuclear interaction 90% width
  _moliere_radii.fill(0.0f);
  const edm::ParameterSet& mconf = conf.getParameterSet("moliereRadii");
  _moliere_radii[0] = mconf.getParameter<double>("HGC_ECAL");  
}

void HGCClusterizer::
buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
	      const std::vector<bool>& rechitMask,
	      const std::vector<bool>& seedable,
	      reco::PFClusterCollection& output) {
  std::vector<bool> usable2d(input->size(),true);
  std::unordered_set<double> unique_depths;
  const reco::PFRecHitCollection& rechits = *input;
  unsigned nseed = 0;

  reco::PFClusterCollection clusters_per_layer;

  for( unsigned i = 0; i < rechits.size(); ++i  ) {
    const auto& hit = rechits[i];
    if(seedable[i] && hit.neighbours().size() > 0 ) {
      ++nseed;     
      /*
      std::cout << "hit " << i << " is seedable with energy "
		<< hit.energy() << "!" << std::endl;
      std::cout << "\thit " << i << " has " << hit.neighbours().size() 
		<< " total neighbours (3D)" << std::endl;
      */
      reco::PFCluster layer_cluster;
      build2DCluster(rechits, rechitMask, seedable,
		     makeRefhit(input,i),usable2d, 
		     layer_cluster);
      unique_depths.insert(std::abs(hit.position().z()));
      
      layer_cluster.setLayer(hit.layer());
      layer_cluster.setSeed(hit.detId());
      _logWeightPosCalc->calculateAndSetPosition(layer_cluster);
      
      if( layer_cluster.hitsAndFractions().size() > 1 ) {
	/*
	std::cout << "\tmade cluster with " 
		<< layer_cluster.hitsAndFractions().size() 
		<< " hits in it!" << std::endl;
	*/
	 clusters_per_layer.push_back(std::move(layer_cluster));
      }
     
      /*
      for( const auto& nbour : hit.neighbours() ) {
	if( nbour->position().z() != hit.position().z() ) {
	  if( seedable[nbour.key()] ) {	    
	    std::cout << "\t\thit " << i 
		      << " seedable neighbour " << nbour.key() 
		      << " in different layer! " 
		      << nbour->position().z() - hit.position().z() 
		      << std::endl;
	    
	  }
	}
      }
      std::cout << "\thit " << i << " has " << hit.neighbours8().size() 
		<< " neighbours in plane (2D)" << std::endl;
      */
      
    } /*else if ( seedable[i] && 
		hit.neighbours().size() == 0 ) {
      std::cout << "hit " << i << " is seedable but has no neighbours "
		<< "and has energy " << hit.energy() << std::endl;
		}*/
  }
  /*
  std::cout << " made " << clusters_per_layer.size() 
	    << " single layer clusters" << std::endl;
  std::cout << " encountered " << unique_depths.size() << " unique_depths." 
	    << std::endl;
  std::cout << " received " << nseed << " seeds!" << std::endl;
  */

  linkClustersInLayer(clusters_per_layer,output); 
}

void HGCClusterizer::build2DCluster(const reco::PFRecHitCollection& input,
				    const std::vector<bool>& rechitMask,
				    const std::vector<bool>& seedable,
				    const reco::PFRecHitRef& current_cell,
				    std::vector<bool>& usable, 
				    reco::PFCluster& cluster) {
  usable[current_cell.key()] = false;
  cluster.addRecHitFraction(reco::PFRecHitFraction(current_cell,1.0));
  
  const reco::PFRecHitRefVector& neighbours_in_layer = 
    current_cell->neighbours8();
  for( const reco::PFRecHitRef& nbour : neighbours_in_layer ) {
    // only cluster if not a seed, not used, and energy less than present
    if( usable[nbour.key()] && !seedable[nbour.key()] && 
	nbour->energy() < current_cell->energy() && 
	rechitMask[nbour.key()]) {
      build2DCluster(input,rechitMask,seedable,nbour,usable,cluster);
    }
  }
}


void HGCClusterizer::
linkClustersInLayer(const reco::PFClusterCollection& input_clusters,
		    reco::PFClusterCollection& output) {  
  std::array<float,3> minpos{ {0.0f,0.0f,0.0f} }, maxpos{ {0.0f,0.0f,0.0f} };
  for( unsigned i = 0 ; i < input_clusters.size(); ++i ) {
    const auto& pos = input_clusters[i].position();
    _nodes.emplace_back(i, (float)pos.X(), (float)pos.Y(), (float)pos.Z());
    if( i == 0 ) {
       minpos[0] = pos.X(); minpos[1] = pos.Y(); minpos[2] = pos.Z();
       maxpos[0] = pos.X(); maxpos[1] = pos.Y(); maxpos[2] = pos.Z();
    } else {
      minpos[0] = std::min((float)pos.X(),minpos[0]);
      minpos[1] = std::min((float)pos.Y(),minpos[1]);
      minpos[2] = std::min((float)pos.Z(),minpos[2]);
      maxpos[0] = std::max((float)pos.X(),maxpos[0]);
      maxpos[1] = std::max((float)pos.Y(),maxpos[1]);
      maxpos[2] = std::max((float)pos.Z(),maxpos[2]);
    }
  }
  KDTreeCube kd_boundingregion(minpos[0],maxpos[0],
			       minpos[1],maxpos[1],
			       minpos[2],maxpos[2]);
  _kdtree.build(_nodes,kd_boundingregion);
  _nodes.clear();
  const float moliere_radius = _moliere_radii[0];
  //const float moliere_radius2 = std::pow(moliere_radius,2.0);
  LinkMap back_links;
  // now link all clusters with in moliere radius for EE (maybe HEF later too)
  for( unsigned i = 0; i < input_clusters.size(); ++i ) {
    const auto& incluster = input_clusters[i];
    const auto& pos = incluster.position();
    auto x = minmax(pos.X()+_moliere_radii[0],pos.X()-_moliere_radii[0]);
    auto y = minmax(pos.Y()+_moliere_radii[0],pos.Y()-_moliere_radii[0]);
    auto z = minmax(pos.Z()+5*_moliere_radii[0],pos.Z()-5*_moliere_radii[0]);
    KDTreeCube kd_searchcube((float)x.first,(float)x.second,
			     (float)y.first,(float)y.second,
			     (float)z.first,(float)z.second);
    _kdtree.search(kd_searchcube,_found);
    for( const auto& found_node : _found ) {
      const auto& found_clus = input_clusters[found_node.data];
      const auto& found_pos  = found_clus.position();
      const auto& diff_pos = found_pos - pos;
      if( diff_pos.rho() < moliere_radius && std::abs(diff_pos.Z()) > 1e-3 ) {
	if( pos.mag2() > found_pos.mag2() ) {
	  back_links.emplace(i,found_node.data);
	} else {
	  back_links.emplace(found_node.data,i);
	}
      }
    }    
    _found.clear();
  }
  // using back-links , use simple metric for now to get something working
  QuickUnion qu(input_clusters.size());  
  unsigned best_match;
  float min_parameter;
  for( unsigned i = 0; i < input_clusters.size(); ++i ) {
    const auto& pos = input_clusters[i].position();
    const auto clusrange = back_links.equal_range(i);
    /*
    std::cout << "cluster " << i << " has " 
	      << std::distance(clusrange.first,clusrange.second) 
	      << " links!" << std::endl;
    */
    min_parameter = std::numeric_limits<float>::max();
    best_match = std::numeric_limits<unsigned>::max();
    for( auto connected = clusrange.first; 
	 connected != clusrange.second; ++connected ) {
      const auto& pos_connected = input_clusters[connected->second].position();
      float angle = (pos_connected - pos).theta();
      if( pos.z() < 0.0f ) angle += M_PI;
      while( angle > M_PI )  angle -= 2*M_PI;
      while( angle < -M_PI ) angle += 2*M_PI;
      angle = std::abs(angle) + 0.001f;
      const float dist2  = (pos_connected - pos).Mag2();
      const float parm  =  dist2*angle*angle;
      if( parm < min_parameter ) {
	best_match = connected->second;
	min_parameter = parm;
      }
    }
    if( best_match != std::numeric_limits<unsigned>::max() ) {
      qu.unite(i,best_match);
    }
  }
  LinkMap merged_clusters;
  UniqueIndices roots;
  for( unsigned i = 0; i < input_clusters.size(); ++i ) {
    const unsigned root = qu.find(i);
    roots.insert(root);
    merged_clusters.emplace(root,i);    
  }
  //std::cout << roots.size() << " final clusters!" << std::endl;
  for( const auto& root : roots ) {
    reco::PFCluster merged_cluster;
    float max_energy = 0;
    reco::PFRecHitRef seed_hit;
    auto range = merged_clusters.equal_range(root);
    for( auto clus = range.first; clus != range.second; ++clus ) {
      const auto& hAndFs = input_clusters[clus->second].recHitFractions();
      for( const auto& hAndF : hAndFs ) {
	merged_cluster.addRecHitFraction(hAndF);
	if( hAndF.recHitRef()->energy() > max_energy ) {
	  max_energy =  hAndF.recHitRef()->energy();
	  seed_hit = hAndF.recHitRef();
	}
      }
    }
    merged_cluster.setSeed(seed_hit->detId());
    merged_cluster.setLayer(seed_hit->layer());
    _pcaPosCalc->calculateAndSetPosition(merged_cluster);
    output.push_back(merged_cluster);
  }
  _found.clear();
  _kdtree.clear();
}
