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

// EM pre-id
#include "RecoEcal/EgammaClusterAlgos/interface/HGCALShowerBasedEmIdentification.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"

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

  struct root_expo {
    root_expo() : constant_(0.0), slope_(0.0), max_radius_(0.0) {}
    root_expo(const double constant, 
	      const double slope,
	      const double max_radius): constant_(constant),
					slope_(slope),
					max_radius_(max_radius) {}
    double  operator()(const int layer) { 
      const double value = 0.1*std::min(std::exp(constant_ + slope_*layer),max_radius_);
      // minimum radius is 1*sqrt(2+epsilon) to make sure first layer forms clusters
      return std::max(1.0*std::sqrt(2.1), value); 
    } 
    double constant_;
    double slope_;
    double max_radius_;
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
  root_expo _em_profile;

  // for track assisted clustering
  const bool _useTrackAssistedClustering;
 
  edm::ESHandle<MagneticField> _bField;
  edm::ESHandle<TrackerGeometry> _tkGeom;
  
  edm::EDGetTokenT<reco::TrackCollection> _tracksToken;
  edm::Handle<reco::TrackCollection> _tracks;
  std::unordered_map<unsigned,unsigned> _rechits_to_clusters;
  std::vector<unsigned> _usable_tracks;

  std::array<std::string,3> _hgc_names;
  std::array<edm::ESHandle<HGCalGeometry>,3> _hgcGeometries;
  std::array<std::vector<ReferenceCountingPointer<BoundDisk> >,3> _plusSurface,_minusSurface;
  std::unique_ptr<PropagatorWithMaterial> _mat_prop;
  
  // coning afterburner
  double _maxClusterAngleToTrack;
  bool _useAfterburner;
  double _minConeAngle, _maxConeAngle, _maxConeDepth;
  unsigned _minECALLayerToCone;

  // EM pre-id
  std::unique_ptr<HGCALShowerBasedEmIdentification> _emPreID;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _emEnergyCalibration;
  // had energy correction
  std::unique_ptr<PFClusterEnergyCorrectorBase> _hadEnergyCalibration;

  // helper functions for various steps in the clustering
  void build2DCluster(const edm::Handle<reco::PFRecHitCollection>&,
		      const reco::PFRecHitCollection&,
		      const std::vector<bool>&,
		      const std::vector<bool>&,
		      const unsigned,
		      const unsigned,
		      std::vector<bool>&,
		      reco::PFCluster&);  

  // linking in Z for clusters
  void linkClustersInLayer(const reco::PFClusterCollection& input_clusters,			   
			   reco::PFClusterCollection& output);

  // use tracks to collect free rechits or associate to clusters after
  // initial EM clustering step (output contains cluster result so far!)
  void trackAssistedClustering(const edm::Handle<reco::PFRecHitCollection>&,
			       const reco::PFRecHitCollection& hits,
			       const std::vector<bool>& rechitMask,
			       std::vector<bool>& rechit_usable,
			       std::vector<bool>& cluster_usable,
			       reco::PFClusterCollection& output);

  // coning afterburner
  void runConingAfterburner(const edm::Handle<reco::PFRecHitCollection>& handle,
			    const reco::PFRecHitCollection& hits,
			    const reco::PFClusterCollection& clusters,
			    const reco::PFCluster& tk_linked_cluster,
			    std::vector<bool>& rechit_usable,
			    std::vector<bool>& cluster_usable,
			    reco::PFCluster& working_cluster);

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
  InitialClusteringStepBase(conf,sumes),  
  _useTrackAssistedClustering(conf.getParameter<bool>("useTrackAssistedClustering")) { 
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
  _moliere_radii[2] = mconf.getParameter<double>("HGC_HCALB");
  const edm::ParameterSet& geoconf = conf.getParameterSet("hgcalGeometryNames");
  _hgc_names[0] = geoconf.getParameter<std::string>("HGC_ECAL");
  _hgc_names[1] = geoconf.getParameter<std::string>("HGC_HCALF");
  _hgc_names[2] = geoconf.getParameter<std::string>("HGC_HCALB");
  //
  const edm::ParameterSet& profile_conf = conf.getParameterSet("emShowerParameterization");
  _em_profile.constant_ = profile_conf.getParameter<double>("HGC_ECAL_constant");
  _em_profile.slope_ = profile_conf.getParameter<double>("HGC_ECAL_slope");
  _em_profile.max_radius_ = profile_conf.getParameter<double>("HGC_ECAL_max_radius");

  // track assisted clustering
  const edm::ParameterSet& tkConf = 
    conf.getParameterSet("trackAssistedClustering");
  // consumes information
  _tracksToken = sumes.consumes<reco::TrackCollection>( tkConf.getParameter<edm::InputTag>("inputTracks") );
  // max allowed angle of cluster to track
  _maxClusterAngleToTrack = tkConf.getParameter<double>("maxClusterAngleToTrack");
  // coning afterburner
  _useAfterburner = tkConf.getParameter<bool>("useAfterburner");
  _minConeAngle = tkConf.getParameter<double>("minConeAngle");
  _maxConeAngle = tkConf.getParameter<double>("maxConeAngle");
  _maxConeDepth = tkConf.getParameter<double>("maxConeDepth");
  _minECALLayerToCone = tkConf.getParameter<unsigned>("minECALLayerToCone");

  // em pre-id
  _emPreID.reset( new HGCALShowerBasedEmIdentification(true) );
  _emPreID->reset();
  const edm::ParameterSet& emEnergyConf = 
    conf.getParameterSet("emEnergyCalibration");
  const std::string& emName = 
    emEnergyConf.getParameter<std::string>("algoName");
  PFClusterEnergyCorrectorBase* emCalib =
    PFClusterEnergyCorrectorFactory::get()->create(emName,emEnergyConf);
  _emEnergyCalibration.reset(emCalib);
  
  // had energy calibration  
  const edm::ParameterSet& hadEnergyConf = 
    conf.getParameterSet("hadEnergyCalibration");
  const std::string& hadName = 
    hadEnergyConf.getParameter<std::string>("algoName");
  PFClusterEnergyCorrectorBase* hadCalib =
    PFClusterEnergyCorrectorFactory::get()->create(hadName,hadEnergyConf);
  _hadEnergyCalibration.reset(hadCalib);
  
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

  // sort seeds by energy
  std::vector<unsigned> seeds;
  for( unsigned i = 0; i < rechits.size(); ++i ) {
    if( seedable[i] ) {
      auto pos = std::lower_bound(seeds.begin(),seeds.end(),i,
				  [&](const unsigned i, const unsigned j) {
				    return ( rechits[i].energy() >= 
					     rechits[j].energy()   );
				  });
      seeds.insert(pos,i);
    }
  }
  
  // get ready for initial topo clustering
  KDTreeCube kd_boundingregion = 
    fill_and_bound_kd_tree(rechits,usable_rechits,_hit_nodes);
  _hit_kdtree.build(_hit_nodes,kd_boundingregion);
  _hit_nodes.clear();
  // make topo-clusters that require that the energy goes
  // down with respect to the last rechit encountered
  // rechits clustered this way are locked from further use  
  for( const unsigned i : seeds ) {
    const auto& hit = rechits[i];
    if(hit.neighbours8().size() > 0 ) {
      reco::PFCluster layer_cluster;
      build2DCluster(input,rechits, rechitMask, seedable,
		     i, i, usable_rechits, 
		     layer_cluster);
      unique_depths.insert(std::abs(hit.position().z()));      
      const auto& hAndFs = layer_cluster.recHitFractions();
      if( hAndFs.size() > 1 ) {
	layer_cluster.setLayer(hit.layer());
	layer_cluster.setSeed(hit.detId());
	_logWeightPosCalc->calculateAndSetPosition(layer_cluster);
	clusters_per_layer.push_back(std::move(layer_cluster));
      } else if ( hAndFs.size() == 1 ) {
	usable_rechits[i] = true;
      }
    }
  }
  _hit_kdtree.clear();
  //std::cout << "layer clusters made: " << clusters_per_layer.size() << std::endl;
  
  reco::PFClusterCollection z_linked_clusters;
  // use topo clusters to link in z
  linkClustersInLayer(clusters_per_layer,z_linked_clusters); 

  // run the em-pre ID on these EM-like clustering result
  // clusters that are EM like are not allowed to be super-clustered
  std::vector<bool> usable_clusters(z_linked_clusters.size(),true);
  std::vector<bool> em_ID_clusters(z_linked_clusters.size(),false);
  for( unsigned i = 0 ; i < z_linked_clusters.size(); ++i ) {
    auto& cluster = z_linked_clusters[i];
    _emPreID->setShowerPosition(cluster.position());
    _emPreID->setShowerDirection(cluster.axis());
    _emEnergyCalibration->correctEnergy(cluster);
    em_ID_clusters[i] = _emPreID->isEm(cluster);
    usable_clusters[i] = !em_ID_clusters[i];
    /*
    if( ! usable_clusters[i] ) { 
      std::cout << "cluster at " << i << " is EM-locked" << std::endl;
    }
    */
    _emPreID->reset();
  }

  // use tracking to clean up unclustered rechits and link clusters  
  if( _useTrackAssistedClustering ) {
    trackAssistedClustering(input,rechits,rechitMask,usable_rechits,
			    usable_clusters,z_linked_clusters);
  }
  
  // stuff usable clusters into the output list
  for( unsigned i = 0; i < z_linked_clusters.size(); ++i ) {
    auto& cluster = z_linked_clusters[i];
    if( cluster.recHitFractions().size() < 5 ) continue;
    if( i >= usable_clusters.size() ) {
      // by definition these are non-EM-like clusters
      _hadEnergyCalibration->correctEnergy(cluster);
      output.push_back(cluster);
    } else if( i < usable_clusters.size() && 
	       (usable_clusters[i] || em_ID_clusters[i]) ) {
      if( !em_ID_clusters[i] ) _hadEnergyCalibration->correctEnergy(cluster);
      output.push_back(cluster);
    }
  }
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
  
  std::sort(_usable_tracks.begin(),_usable_tracks.end(),
	    [&](const unsigned i, const unsigned j) {
	      return tracks[i].pt() > tracks[j].pt();
	    });
  
}

void HGCClusterizer::build2DCluster(const edm::Handle<reco::PFRecHitCollection>& handle,
				    const reco::PFRecHitCollection& input,
				    const std::vector<bool>& rechitMask,
				    const std::vector<bool>& seedable,
				    const unsigned seed_index,
				    const unsigned current_index,
				    std::vector<bool>& usable, 
				    reco::PFCluster& cluster) {
  const reco::PFRecHit& current_cell = input[current_index];
  
  double moliere_radius = -1.0;
  const math::XYZPoint pos = current_cell.position();
  DetId cellid = current_cell.detId();
  switch( current_cell.layer() ) {
  case PFLayer::HGC_ECAL:
    moliere_radius = _em_profile(HGCEEDetId(cellid).layer());
    break;
  case PFLayer::HGC_HCALF:
    moliere_radius = _moliere_radii[1];
    break;
  case PFLayer::HGC_HCALB:
    moliere_radius = _moliere_radii[2];
    break;
  default:
    break;
  }
  
  auto x_rh = minmax(pos.x()+moliere_radius,pos.x()-moliere_radius);
  auto y_rh = minmax(pos.y()+moliere_radius,pos.y()-moliere_radius);
  auto z_rh = minmax(pos.z()+1e-3,pos.z()-1e-3);
  KDTreeCube hit_searchcube((float)x_rh.first,(float)x_rh.second,
			    (float)y_rh.first,(float)y_rh.second,
			    (float)z_rh.first,(float)z_rh.second);
  std::vector<KDNode> found;
  _hit_kdtree.search(hit_searchcube,found);
  // check to see if this hit is closer to another seed than the original
  const auto& seed_position = input[seed_index].position();
  const auto& dist_to_seed = (pos - seed_position);
  math::XYZVector dist_to_nearest_seed = dist_to_seed;
  for( const KDNode& nbourpoint : found ) {    
    const auto& nbpos = input[nbourpoint.data].position();
    const auto& dist_to_nb = (pos - nbpos);
    if( seedable[nbourpoint.data] && nbourpoint.data != seed_index) {
      if( dist_to_nb.mag2() < dist_to_nearest_seed.mag2() ) {
	dist_to_nearest_seed = dist_to_nb;
      }  
    }
  }
  if( dist_to_nearest_seed.mag2() < dist_to_seed.mag2() ) return;
  // set hit as used
  usable[current_index] = false;  
  cluster.addRecHitFraction(reco::PFRecHitFraction(makeRefhit(handle,current_index),1.0));

  for( const KDNode& nbourpoint : found ) {
    // only cluster if not a seed, not used, and energy less than present
    const reco::PFRecHit& nbour = input[nbourpoint.data];
    if( usable[nbourpoint.data] && !seedable[nbourpoint.data] && 
	nbour.energy() <= current_cell.energy() && // <= takes care of MIP sea
	rechitMask[nbourpoint.data] &&
	(nbour.position() - current_cell.position()).mag2() < moliere_radius*moliere_radius) {
      build2DCluster(handle,input,rechitMask,seedable,seed_index,nbourpoint.data,usable,cluster);
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
    float z_separation = -1.0;
    DetId seedid = input_clusters[i].seed();
    switch( seedid.subdetId() ) {
    case HGCEE:
      moliere_radius = std::max((double)_moliere_radii[0],_em_profile(HGCEEDetId(seedid).layer()));
      z_separation = 2.5;
      break;
    case HGCHEF:
      moliere_radius = _moliere_radii[1];
      z_separation = moliere_radius;
      break;
    case HGCHEB:
      moliere_radius = _moliere_radii[2];
      z_separation = moliere_radius;
      break;
    default: 
      break;
    }
    const auto& incluster = input_clusters[i];
    const auto& pos = incluster.position();
    auto x = minmax(pos.X()+moliere_radius,pos.X()-moliere_radius);
    auto y = minmax(pos.Y()+moliere_radius,pos.Y()-moliere_radius);
    auto z = minmax(pos.Z()+2.0*moliere_radius,pos.Z()-2.0*moliere_radius);
    KDTreeCube kd_searchcube((float)x.first,(float)x.second,
			     (float)y.first,(float)y.second,
			     (float)z.first,(float)z.second);
    _cluster_kdtree.search(kd_searchcube,_found);
    for( const auto& found_node : _found ) {
      const auto& found_clus = input_clusters[found_node.data];
      const auto& found_pos  = found_clus.position();
      const auto& diff_pos = found_pos - pos;
      const double angle = std::acos(diff_pos.rho()/diff_pos.r());
      if( angle > 0.2 && diff_pos.r() < moliere_radius && 
	  std::abs(diff_pos.z()) < z_separation && 
	  std::abs(diff_pos.Z()) > 1e-3 ) {
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
  _rechits_to_clusters.clear();
  unsigned iclus = 0;
  for( const auto& root : roots ) {
    reco::PFCluster merged_cluster;
    float max_energy = 0;
    reco::PFRecHitRef seed_hit;
    auto range = merged_clusters.equal_range(root);
    for( auto clus = range.first; clus != range.second; ++clus ) {
      const auto& hAndFs = input_clusters[clus->second].recHitFractions();
      for( const auto& hAndF : hAndFs ) {
	merged_cluster.addRecHitFraction(hAndF);
	_rechits_to_clusters.emplace(hAndF.recHitRef().key(),iclus);
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
    ++iclus;
  }
  _found.clear();
  _cluster_kdtree.clear();
}

void HGCClusterizer::
trackAssistedClustering(const edm::Handle<reco::PFRecHitCollection>& hits_handle,
			const reco::PFRecHitCollection& hits,
			const std::vector<bool>& rechitMask,
			std::vector<bool>& rechit_usable,
			std::vector<bool>& cluster_usable,
			reco::PFClusterCollection& output) {
  // setup kd tree searches for clusters and tracks
  // setup clusters
  _found.clear();
  KDTreeCube cluster_bounds = 
    fill_and_bound_kd_tree(output,
			   cluster_usable,
			   _cluster_nodes);  
  _cluster_kdtree.build(_cluster_nodes,cluster_bounds);
  _cluster_nodes.clear();
  // setup rechits
  KDTreeCube hit_bounds = 
    fill_and_bound_kd_tree(hits,
			   rechitMask,
			   _hit_nodes);
  _hit_kdtree.build(_hit_nodes,hit_bounds);
  _hit_nodes.clear();
  
  const reco::TrackCollection& tracks = *_tracks;
  //std::cout << "there are " << _usable_tracks.size() << " tracks to process!" << std::endl;
  for( const unsigned i : _usable_tracks ) {
    std::unordered_map<unsigned,unsigned> hits_of_cluster_on_track;
    std::unordered_map<unsigned,bool> clusters_in_track;
    reco::PFCluster temp;
    const reco::Track& tk = tracks[i];
    //std::cout << "got track: " << tk.pt() << ' ' << tk.eta() << ' ' << tk.phi() << std::endl;
    const TrajectoryStateOnSurface myTSOS = trajectoryStateTransform::outerStateOnSurface(tk, *(_tkGeom.product()),_bField.product());
    auto detbegin = myTSOS.globalPosition().z() > 0 ? _plusSurface.begin() : _minusSurface.begin();
    auto detend = myTSOS.globalPosition().z() > 0 ? _plusSurface.end() : _minusSurface.end();
    for( auto det = detbegin; det != detend; ++det ) {      
      for( const auto& layer : *det ) {
	_found.clear();
	TrajectoryStateOnSurface piStateAtSurface = _mat_prop->propagate(myTSOS, *layer);
	if( piStateAtSurface.isValid() ) {
	  GlobalPoint pt = piStateAtSurface.globalPosition();
	  math::XYZPoint tkpos(pt.x(),pt.y(),pt.z());
	  GlobalError xyzerr = piStateAtSurface.cartesianError().position();
	  // just take maximal error envelope
	  const float xyerr = std::sqrt(xyzerr.cxx() + xyzerr.cyy() + std::abs(xyzerr.cyx()) ); 
	  const float search_radius = std::max( xyerr, 2.0f ); // use 2.0 cm as minimum matching distance for now
	  /*
	  std::cout << "\ttrack successfully got to: " 
		    << pt.x() << " +/- " <<  std::sqrt(xyzerr.cxx()) << ' ' 
		    << pt.y() << " +/- "<< ' ' << std::sqrt(xyzerr.cyy()) << ' ' 
		    << pt.z() << " +/- " << std::sqrt(xyzerr.czz()) << " searching in " << search_radius << std::endl;	  
	  */
	  // look for nearby un-clustered rechits and add them to the track
	  auto x_rh = minmax(pt.x()+search_radius,pt.x()-search_radius);
	  auto y_rh = minmax(pt.y()+search_radius,pt.y()-search_radius);
	  auto z_rh = minmax(pt.z()+1e-2,pt.z()-1e-2);
	  KDTreeCube rechit_searchcube((float)x_rh.first,(float)x_rh.second,
				       (float)y_rh.first,(float)y_rh.second,
				       (float)z_rh.first,(float)z_rh.second);
	  _hit_kdtree.search(rechit_searchcube,_found);
	  double least_distance = std::numeric_limits<double>::max();
	  unsigned best_index = std::numeric_limits<unsigned>::max();
	  for( const auto& hit : _found ) {
	    const auto& pos = hits[hit.data].position();	  
	    double dr = (tkpos - pos).r();
	    if( dr < search_radius && dr < least_distance )  {	      
	      best_index = hit.data;
	      least_distance = dr;
	    }
	  }
	  if( least_distance != std::numeric_limits<double>::max() ) {
	    if( rechit_usable[best_index] ) {
	      rechit_usable[best_index] = false; // do not allow other tracks to grab this
	      //const auto& pos = hits[best_index].position();
	      temp.addRecHitFraction(reco::PFRecHitFraction(makeRefhit(hits_handle,best_index),1.0));
	      //std::cout << "adding hit at: (" << pos.x() << ',' << pos.y() << ',' << pos.z() << ") to cluster! (least distance = " << least_distance << " cm)" << std::endl;	    
	    } else { // rechit is in a cluster or masked
	      auto cluster_match = _rechits_to_clusters.find(best_index);
	      if( cluster_match != _rechits_to_clusters.end() ) {
		if( hits_of_cluster_on_track.find(cluster_match->second) == 
		    hits_of_cluster_on_track.end() ) {
		  hits_of_cluster_on_track[cluster_match->second] = 0;
		}
		hits_of_cluster_on_track[cluster_match->second] += 1;
		const GlobalVector& tkdir_orig = piStateAtSurface.globalDirection();
		math::XYZVector tkdir(tkdir_orig.x(),tkdir_orig.y(),tkdir_orig.z());
		const math::XYZVector& clusdir = output[cluster_match->second].axis();
		// get angle between vectors...
		const double angle = std::abs(std::acos(tkdir.Dot(clusdir)/std::sqrt(tkdir.mag2()*clusdir.mag2())));
		//const auto& pos = output[cluster_match->second].position();		
		if( cluster_usable[cluster_match->second] && 
		    ( angle < _maxClusterAngleToTrack || 
		      hits_of_cluster_on_track[cluster_match->second] > 7 ||
		      output[cluster_match->second].recHitFractions().size() < 10 ) ) { 
		  clusters_in_track[cluster_match->second] = true;
		  cluster_usable[cluster_match->second] = false;
		  for( const auto& hAndF : output[cluster_match->second].recHitFractions() ) {
		    temp.addRecHitFraction(hAndF);
		  }
		  if( _useAfterburner ) {
		    runConingAfterburner(hits_handle,
					 hits,
					 output,
					 output[cluster_match->second],
					 rechit_usable,
					 cluster_usable,
					 temp);
		  }
		  /*
		  std::cout << "adding cluster at: (" << pos.x() << ',' << pos.y() << ',' << pos.z() << ") " 
			    <<  output[cluster_match->second].pt() << ' ' << pos.eta() << " to had-supercluster! nhits = " 
			    <<  hits_of_cluster_on_track[cluster_match->second] << std::endl;
		  */
		}/*  else {
		  
		  std::cout << "rejected cluster at : (" << pos.x() << ',' << pos.y() << ',' << pos.z() << ") nhits = " 
			    << hits_of_cluster_on_track[cluster_match->second] << " pt = " 
			    <<  output[cluster_match->second].pt() << " eta = "  
			    << pos.eta() << " to had-supercluster! angle = "
			    << angle << " posdiff = "<< (pos - tkpos).rho()
			    << " cluster_usable = " << cluster_usable[cluster_match->second];
		  if( !cluster_usable[cluster_match->second] ) {
		    _emPreID->reset();
		    _emPreID->setShowerPosition(output[cluster_match->second].position());
		    _emPreID->setShowerDirection(output[cluster_match->second].axis());
		    if( _emPreID->isEm(output[cluster_match->second]) ) {
		      std::cout << " because cluster is EM!";
		    } else {
		      std::cout << " because cluster already used!";		      
		    }
		  }
		  if( angle >= _maxClusterAngleToTrack && (pos - tkpos).rho() >= 1.5) {
		    std::cout << " because cluster doesn't point along track! "; 
		  }
		  std::cout << std::endl;
		  
		  } */
		
	      }
	    }
	  }
	  _found.clear();	  
	}
      }
    }
    if( temp.recHitFractions().size() ) {
      const auto& seed_hit = temp.recHitFractions()[0].recHitRef();
      temp.setSeed(seed_hit->detId());
      temp.setLayer(seed_hit->layer());
      if( temp.recHitFractions().size() > 1 ) {
	_pcaPosCalc->calculateAndSetPosition(temp); 
	// set axis to track trajectory at seed layer?
      } else { // give a reasonable axis guess.. point it at 0,0,0
	_logWeightPosCalc->calculateAndSetPosition(temp);
	const auto& pos = temp.position();
	const auto& tempaxis = pos/pos.r();
	math::XYZVector axis(tempaxis.x(),tempaxis.y(),tempaxis.z());
	temp.setAxis(axis);
	temp.calculatePositionREP();
      }
      output.push_back(temp);
    }
  }
  _hit_kdtree.clear();
  _cluster_kdtree.clear();
}

void HGCClusterizer::
runConingAfterburner(const edm::Handle<reco::PFRecHitCollection>& handle,
		     const reco::PFRecHitCollection& hits,
		     const reco::PFClusterCollection& clusters,
		     const reco::PFCluster& tk_linked_cluster,
		     std::vector<bool>& rechit_usable,
		     std::vector<bool>& cluster_usable,
		     reco::PFCluster& working_cluster) {
  typedef ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double>, ROOT::Math::DefaultCoordinateSystemTag> PolarPoint;
  std::vector<KDNode> found;
  // calculate the bounding cylinder for the cone at this depth
  // first, get the minimum z of the cluster being used to position the cone
  std::vector<unsigned> hits_input;
  const auto& rhfs = tk_linked_cluster.recHitFractions();
  for( unsigned k = 0; k < rhfs.size(); ++k ) {    
    auto pos = std::lower_bound(hits_input.begin(),hits_input.end(),k,
				[&](const unsigned i, const unsigned j) {
				  return ( std::abs(hits[i].position().z()) < 
					   std::abs(hits[rhfs[j].recHitRef().key()].position().z())   );
				});
    hits_input.insert(pos,rhfs[k].recHitRef().key());
  }  
  // get all the hits at the same depth
  auto hits_first_depth = std::equal_range(hits_input.begin(),
					   hits_input.end(),
					   hits_input[0],
					   [&](const unsigned i, const unsigned j) {
					     return ( std::abs(hits[i].position().z()) < 
						      std::abs(hits[j].position().z())   );
					   });
  const math::XYZVector& cone_axis = tk_linked_cluster.axis();
  math::XYZPoint cone_vertex_cartesian = tk_linked_cluster.position();
  PolarPoint cone_vertex_polar;
  cone_vertex_polar.SetXYZ(cone_vertex_cartesian.x(),
			   cone_vertex_cartesian.y(),
			   cone_vertex_cartesian.z());
  while( std::abs(cone_vertex_polar.z()) > 
	 std::abs(hits[*hits_first_depth.first].position().z()) ) {
    cone_vertex_polar = cone_vertex_polar - 10.0*cone_axis.Unit();
  }
  cone_vertex_cartesian.SetXYZ(cone_vertex_polar.x(),
			       cone_vertex_polar.y(),
			       cone_vertex_polar.z());
  //std::cout << "input cluster barycenter: " << tk_linked_cluster.position() << std::endl;
  //std::cout << "got initial cone vertex position: " << cone_vertex_cartesian << std::endl;
  // see if the already built cluster exists within the cone as defined by default
  double max_hit_angle = 0.0;
  //unsigned max_angle_index = std::numeric_limits<unsigned>::max();
  for(unsigned i = 0; i < hits_input.size(); ++i ) {
    double angle = (cone_vertex_cartesian - hits[hits_input[i]].position()).theta();
    if( cone_vertex_cartesian.z() > 0.0f ) angle += M_PI;
    while( angle > M_PI )  angle -= 2*M_PI;
    while( angle < -M_PI ) angle += 2*M_PI;
    if( std::abs(angle) > std::abs(max_hit_angle) ) {
      max_hit_angle = angle;
      //max_angle_index = i;
    }
  }
  /*
  std::cout << "index of max angle: " << max_angle_index << std::endl;
  std::cout << "maximum angle to cone vertex: " << max_hit_angle << std::endl;
  std::cout << "cone angle from python: " << _minConeAngle << std::endl;
  */
  max_hit_angle = std::min(std::abs(max_hit_angle),_maxConeAngle); // restrict to one radian
  if( max_hit_angle <= _maxConeAngle ) { // less than 1 radian, get rid of crazy
    // if we have a good angle, create the KD tree bounding region using
    // the cone radius at max cone depth
    double radius = _maxConeDepth*std::tan(std::abs(max_hit_angle));
    double signedConeDepth = ( cone_vertex_cartesian.z() > 0 ? _maxConeDepth : -_maxConeDepth );
    auto x_rh = minmax(cone_vertex_cartesian.x()+radius,
		       cone_vertex_cartesian.x()-radius);
    auto y_rh = minmax(cone_vertex_cartesian.y()+radius,
		       cone_vertex_cartesian.y()-radius);
    auto z_rh = minmax(cone_vertex_cartesian.z(),
		       cone_vertex_cartesian.z()+signedConeDepth);
    KDTreeCube rechit_searchcube((float)x_rh.first,(float)x_rh.second,
				 (float)y_rh.first,(float)y_rh.second,
				 (float)z_rh.first,(float)z_rh.second);
    _hit_kdtree.search(rechit_searchcube,found);
    unsigned added_hits = 0;
    for( const auto& fhit : found ) {
      if( !rechit_usable[fhit.data] ) continue;
      double rhangle = (cone_vertex_cartesian - hits[fhit.data].position()).theta();
      if( cone_vertex_cartesian.z() > 0.0f ) rhangle += M_PI;
      while( rhangle >  M_PI ) rhangle -= 2*M_PI;
      while( rhangle < -M_PI ) rhangle += 2*M_PI;
      if( std::abs(rhangle) < std::abs(max_hit_angle) && 
	  std::abs(hits[fhit.data].position().z()) < std::abs(cone_vertex_cartesian.z())+_maxConeDepth ) {
	rechit_usable[fhit.data] = false;
	working_cluster.addRecHitFraction(reco::PFRecHitFraction(makeRefhit(handle,fhit.data),1.0));
	++added_hits;
      }
    }
    found.clear();
    _cluster_kdtree.search(rechit_searchcube,found);
    unsigned added_clusters(0);
    for( const auto& fcluster : found ) {
      const auto& thecluster = clusters[fcluster.data];
      if( !cluster_usable[fcluster.data] ) continue;
      if( &thecluster == &tk_linked_cluster ) continue; // don't repeatedly add the same cluster
      double clusangle = (cone_vertex_cartesian - thecluster.position()).theta();
      if( cone_vertex_cartesian.z() > 0.0f ) clusangle += M_PI;
      while( clusangle >  M_PI ) clusangle -= 2*M_PI;
      while( clusangle < -M_PI ) clusangle += 2*M_PI;
      if( std::abs(clusangle) < std::abs(max_hit_angle) && 
	      std::abs(thecluster.position().z()) < std::abs(cone_vertex_cartesian.z())+_maxConeDepth ) {
	cluster_usable[fcluster.data] = false;
	for( const auto& hit : thecluster.recHitFractions() ) {
	  working_cluster.addRecHitFraction(hit);  
	  ++added_hits;
	}
	++added_clusters;
      }    
    }
  }
}
