#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"

// for track propagation through HGC
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/FCalGeometry/interface/HGCalGeometry.h"

//geometry records
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/Framework/interface/Event.h"

#include<unordered_map>

// helpful tools
#include "KDTreeLinkerAlgoT.h"
#include <unordered_map>
#include <unordered_set>

//local tools
namespace {
  std::pair<float,float> minmax(const float a, const float b) {
    return ( b < a ? 
	     std::pair<float,float>(b, a) : 
	     std::pair<float,float>(a, b)   );
  }
  
  template<typename T>
  KDTreeCube fill_and_bound_kd_tree(const std::vector<T>& points,
				    const std::vector<bool>& usable,
				    std::vector<KDTreeNodeInfoT<unsigned,3> >& nodes) {
    std::array<float,3> minpos{ {0.0f,0.0f,0.0f} }, maxpos{ {0.0f,0.0f,0.0f} };
    for( unsigned i = 0 ; i < points.size(); ++i ) {
      if( !usable[i] ) continue;
      const auto& pos = points[i].position();
      nodes.emplace_back(i, (float)pos.X(), (float)pos.Y(), (float)pos.Z());
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
    return KDTreeCube(minpos[0],maxpos[0],
		      minpos[1],maxpos[1],
		      minpos[2],maxpos[2]);
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

//class def
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
  HGCClusterizer(const edm::ParameterSet& conf,
		 edm::ConsumesCollector& sumes);
  virtual ~HGCClusterizer() {}
  HGCClusterizer(const B2DGT&) = delete;
  B2DGT& operator=(const B2DGT&) = delete;

  virtual void update(const edm::EventSetup& es) override final;
  
  virtual void updateEvent(const edm::Event& ev) override final;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
		     const std::vector<bool>&,
		     const std::vector<bool>&, 
		     reco::PFClusterCollection&);
  
private:
  // used for rechit searching 
  std::vector<KDNode> _cluster_nodes, _hit_nodes, _found;
  KDTree _cluster_kdtree,_hit_kdtree;

  std::unique_ptr<PosCalc> _logWeightPosCalc,_pcaPosCalc;

  std::array<float,3> _moliere_radii;

  // for track assisted clustering
  edm::ESHandle<MagneticField> _bField;
  edm::ESHandle<TrackerGeometry> _tkGeom;
  
  edm::EDGetTokenT<reco::TrackCollection> _tracksToken;
  edm::Handle<reco::TrackCollection> _tracks;
  std::vector<unsigned> _usable_tracks;

  std::array<std::string,3> _hgc_names;
  std::array<edm::ESHandle<HGCalGeometry>,3> _hgcGeometries;
  std::array<std::vector<ReferenceCountingPointer<BoundDisk> >,3> _plusSurface,_minusSurface;
  std::unique_ptr<PropagatorWithMaterial> _mat_prop;
  

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

  // use tracks to collect free rechits or associate to clusters after
  // initial EM clustering step (output contains cluster result so far!)
  void trackAssistedClustering(const reco::PFRecHitCollection& hits,
			       std::vector<bool>& rechit_usable,
			       std::vector<bool>& cluster_usable,
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

HGCClusterizer::HGCClusterizer(const edm::ParameterSet& conf,
			       edm::ConsumesCollector& sumes) :
  InitialClusteringStepBase(conf,sumes) { 
  // clean initial state for searching
  _cluster_nodes.clear(); _found.clear(); _cluster_kdtree.clear();
  _hit_nodes.clear(); _hit_kdtree.clear();
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
  
  // get moliere radius, nuclear interaction 90% width, geometry names
  _moliere_radii.fill(0.0f);
  const edm::ParameterSet& mconf = conf.getParameterSet("moliereRadii");
  _moliere_radii[0] = mconf.getParameter<double>("HGC_ECAL");  
  _moliere_radii[1] = mconf.getParameter<double>("HGC_HCALF");
  const edm::ParameterSet& geoconf = conf.getParameterSet("hgcalGeometryNames");
  _hgc_names[0] = geoconf.getParameter<std::string>("HGC_ECAL");
  _hgc_names[1] = geoconf.getParameter<std::string>("HGC_HCALF");
  _hgc_names[2] = geoconf.getParameter<std::string>("HGC_HCALB");

  // track assisted clustering
  const edm::ParameterSet& tkConf = 
    conf.getParameterSet("trackAssistedClustering");
  // consumes information
  _tracksToken = sumes.consumes<reco::TrackCollection>( tkConf.getParameter<edm::InputTag>("inputTracks") );
  
}

void HGCClusterizer::
buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
	      const std::vector<bool>& rechitMask,
	      const std::vector<bool>& seedable,
	      reco::PFClusterCollection& output) {
  std::vector<bool> usable_rechits(input->size(),true);
  std::unordered_set<double> unique_depths;
  const reco::PFRecHitCollection& rechits = *input;

  reco::PFClusterCollection clusters_per_layer;

  // make topo-clusters that require that the energy goes
  // down with respect to the last rechit encountered
  // rechits clustered this way are locked from further use
  for( unsigned i = 0; i < rechits.size(); ++i  ) {
    const auto& hit = rechits[i];
    if(seedable[i] && hit.neighbours().size() > 0 ) {
      reco::PFCluster layer_cluster;
      build2DCluster(rechits, rechitMask, seedable,
		     makeRefhit(input,i),usable_rechits, 
		     layer_cluster);
      unique_depths.insert(std::abs(hit.position().z()));
      
      layer_cluster.setLayer(hit.layer());
      layer_cluster.setSeed(hit.detId());
      _logWeightPosCalc->calculateAndSetPosition(layer_cluster);
      
      if( layer_cluster.hitsAndFractions().size() > 1 ) {
	 clusters_per_layer.push_back(std::move(layer_cluster));
      }      
    }
  }
  // use topo clusters to link in z
  linkClustersInLayer(clusters_per_layer,output); 

  // use tracking to clean up unclustered rechits
  std::vector<bool> usable_clusters(output.size(),true);
  trackAssistedClustering(rechits,usable_rechits,
			  usable_clusters,output);
}

void HGCClusterizer::update(const edm::EventSetup& es) {
  constexpr float m_pion = 0.1396;
  // get dependencies for setting up propagator  
  es.get<IdealMagneticFieldRecord>().get(_bField);
  es.get<TrackerDigiGeometryRecord>().get(_tkGeom);
  // get HGC geometries (assume that layers are ordered in Z!)
  for( unsigned i = 0; i < _hgcGeometries.size(); ++i ) {
    es.get<IdealGeometryRecord>().get(_hgc_names[i],_hgcGeometries[i]);
  }

  // make propagator
  _mat_prop.reset( new PropagatorWithMaterial(alongMomentum, m_pion, _bField.product()) );
  // setup HGC layers for track propagation
  Surface::RotationType rot; //unit rotation matrix
  for( unsigned i = 0; i < _hgcGeometries.size(); ++i ) {
    _minusSurface[i].clear();
    _plusSurface[i].clear();
    const HGCalDDDConstants &dddCons=_hgcGeometries[i]->topology().dddConstants();
    std::map<float,float> zrhoCoord;
    auto firstLayerIt = dddCons.getFirstTrForm();
    auto lastLayerIt = dddCons.getLastTrForm();
    for(auto layerIt=firstLayerIt; layerIt !=lastLayerIt; layerIt++) {
      float Z(fabs(layerIt->h3v.z()));
      float Radius(dddCons.getLastModule(true)->tl+layerIt->h3v.perp());
      zrhoCoord[Z]=Radius;
    }
    for(auto it=zrhoCoord.begin(); it != zrhoCoord.end(); it++) {
      float Z(it->first);
      float Radius(it->second);
      _minusSurface[i].push_back(ReferenceCountingPointer<BoundDisk> ( new BoundDisk( Surface::PositionType(0,0,-Z), rot, new SimpleDiskBounds( 0, Radius, -0.001, 0.001))));
      _plusSurface[i].push_back(ReferenceCountingPointer<BoundDisk> ( new BoundDisk( Surface::PositionType(0,0,+Z), rot, new SimpleDiskBounds( 0, Radius, -0.001, 0.001))));
    }    
  }  
}
  
void HGCClusterizer::updateEvent(const edm::Event& ev) {
  _usable_tracks.clear();
  ev.getByToken(_tracksToken,_tracks);
  const reco::TrackCollection tracks = *_tracks;  
  _usable_tracks.reserve(tracks.size());
  for( unsigned i = 0; i < tracks.size(); ++i ) {
    const reco::Track& tk = tracks[i];
    const double tk_abs_eta = std::abs(tk.eta());
    // require track eta within fiducial HGC volume
    bool usable = ( tk_abs_eta > 1.45 && tk_abs_eta < 3.0 ); ;
    if( usable ) _usable_tracks.push_back(i);
  }
  _usable_tracks.shrink_to_fit();
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
  std::vector<bool> dummy(input_clusters.size(),true);  
  KDTreeCube kd_boundingregion = 
    fill_and_bound_kd_tree(input_clusters,dummy,_cluster_nodes);
  _cluster_kdtree.build(_cluster_nodes,kd_boundingregion);
  _cluster_nodes.clear();
  
  //const float moliere_radius2 = std::pow(moliere_radius,2.0);
  LinkMap back_links;
  // now link all clusters with in moliere radius for EE + HEF
  for( unsigned i = 0; i < input_clusters.size(); ++i ) {
    float moliere_radius = -1.0;
    switch( input_clusters[i].seed().subdetId() ) {
    case HGCEE:
      moliere_radius = _moliere_radii[0];
      break;
    case HGCHEF:
    case HGCHEB:
      moliere_radius = _moliere_radii[1];
      break;
    default: 
      break;
    }
    const auto& incluster = input_clusters[i];
    const auto& pos = incluster.position();
    auto x = minmax(pos.X()+moliere_radius,pos.X()-moliere_radius);
    auto y = minmax(pos.Y()+moliere_radius,pos.Y()-moliere_radius);
    auto z = minmax(pos.Z()+2*moliere_radius,pos.Z()-2*moliere_radius);
    KDTreeCube kd_searchcube((float)x.first,(float)x.second,
			     (float)y.first,(float)y.second,
			     (float)z.first,(float)z.second);
    _cluster_kdtree.search(kd_searchcube,_found);
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
    // only use the PCA if there *is* z extent to the cluster
    if( std::distance(range.first,range.second) > 1 ) {
      _pcaPosCalc->calculateAndSetPosition(merged_cluster); 
    } else { // give a reasonable axis guess.. point it at 0,0,0
      _logWeightPosCalc->calculateAndSetPosition(merged_cluster);
      const auto& pos = merged_cluster.position();
      const auto& tempaxis = pos/pos.r();
      math::XYZVector axis(tempaxis.x(),tempaxis.y(),tempaxis.z());
      merged_cluster.setAxis(axis);
      merged_cluster.calculatePositionREP();
    }
    output.push_back(merged_cluster);    
  }
  _found.clear();
  _cluster_kdtree.clear();
}

void HGCClusterizer::
trackAssistedClustering(const reco::PFRecHitCollection& hits,
			std::vector<bool>& rechit_usable,
			std::vector<bool>& cluster_usable,
			reco::PFClusterCollection& output) {
  // setup kd tree searches for clusters and tracks
  // setup clusters
  KDTreeCube cluster_bounds = 
    fill_and_bound_kd_tree(output,
			   cluster_usable,
			   _cluster_nodes);  
  _cluster_kdtree.build(_cluster_nodes,cluster_bounds);
  // setup rechits
  KDTreeCube hit_bounds = 
    fill_and_bound_kd_tree(hits,
			   rechit_usable,
			   _hit_nodes);
  _hit_kdtree.build(_hit_nodes,hit_bounds);
  
  const reco::TrackCollection& tracks = *_tracks;
  for( const unsigned i : _usable_tracks ) {
    const reco::Track& tk = tracks[i];
    std::cout << "got track: " << tk.pt() << ' ' << tk.eta() << ' ' << tk.phi() << std::endl;
    const TrajectoryStateOnSurface myTSOS = trajectoryStateTransform::outerStateOnSurface(tk, *(_tkGeom.product()),_bField.product());
    auto detbegin = myTSOS.globalPosition().z() > 0 ? _plusSurface.begin() : _minusSurface.begin();
    auto detend = myTSOS.globalPosition().z() > 0 ? _plusSurface.end() : _minusSurface.end();
    for( auto det = detbegin; det != detend; ++det ) {      
      for( const auto& layer : *det ) {
	TrajectoryStateOnSurface piStateAtSurface = _mat_prop->propagate (myTSOS, *layer);
	if( piStateAtSurface.isValid() ) {
	  GlobalPoint pt = piStateAtSurface.globalPosition();
	  GlobalError xyzerr = piStateAtSurface.cartesianError().position();
	  // just take maximal error envelope
	  const float xyerr = std::sqrt(xyzerr.cxx() + xyzerr.cyy() + std::abs(xyzerr.cyx()) ); 
	  const float search_radius = std::max( xyerr, 1.5f ); // use 1.5 cm as minimum matching distance for now
	  std::cout << "\ttrack successfully got to: " 
		    << pt.x() << " +/- " <<  std::sqrt(xyzerr.cxx()) << ' ' 
		    << pt.y() << " +/- "<< ' ' << std::sqrt(xyzerr.cyy()) << ' ' 
		    << pt.z() << " +/- " << std::sqrt(xyzerr.czz()) << " searching in " << search_radius << std::endl;
	  
	  
	}
      }
    }
  }  
}
