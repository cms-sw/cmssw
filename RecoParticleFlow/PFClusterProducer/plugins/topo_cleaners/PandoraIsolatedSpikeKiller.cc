#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"

#include "KDTreeLinkerAlgoT.h"

#include <unordered_map>

class PandoraIsolatedSpikeKiller : public RecHitTopologicalCleanerBase {
  typedef KDTreeLinkerAlgo<unsigned,3> KDTree;
  typedef KDTreeNodeInfoT<unsigned,3> KDNode;
 public:
  
  struct spike_cleaning {
    double _singleSpikeThresh;
    double _minS4S1_a;
    double _minS4S1_b;
    double _doubleSpikeS6S2;
    double _eneThreshMod;
    double _fracThreshMod;
    double _doubleSpikeThresh;
  };

  PandoraIsolatedSpikeKiller(const edm::ParameterSet& conf);
  PandoraIsolatedSpikeKiller(const PandoraIsolatedSpikeKiller&) = delete;
  PandoraIsolatedSpikeKiller& operator=(const PandoraIsolatedSpikeKiller&) = delete;

  void clean( const edm::Handle<reco::PFRecHitCollection>& input,
	      std::vector<bool>& mask );

 private:
  // can simplify from pseudo layers to simply searching in a cylinder
  const double _hit_search_radius, _hit_search_length, _weight_power, 
    _weight_cut; 
  // used for rechit searching 
  std::vector<KDNode> _hit_nodes, _found;
  KDTree _hit_kdtree;
  
};

DEFINE_EDM_PLUGIN(RecHitTopologicalCleanerFactory,
		  PandoraIsolatedSpikeKiller,"PandoraIsolatedSpikeKiller");

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
}

PandoraIsolatedSpikeKiller::
PandoraIsolatedSpikeKiller(const edm::ParameterSet& conf) :
  RecHitTopologicalCleanerBase(conf),
  _hit_search_radius(conf.getParameter<double>("hit_search_radius")),
  _hit_search_length(conf.getParameter<double>("hit_search_length")),
  _weight_power(conf.getParameter<double>("weight_power")),
  _weight_cut(conf.getParameter<double>("weight_cut")) {  
  // ensure reasonable starting conditions
  _found.clear();
  _hit_nodes.clear();
  _hit_kdtree.clear();
}


void PandoraIsolatedSpikeKiller::
clean(const edm::Handle<reco::PFRecHitCollection>& input,
      std::vector<bool>& mask ) {
  const reco::PFRecHitCollection& prod = *input;
  // setup kd-tree search region 
  KDTreeCube kd_boundingregion = 
    fill_and_bound_kd_tree(prod,mask,_hit_nodes);
  _hit_kdtree.build(_hit_nodes,kd_boundingregion);
  _hit_nodes.clear();
  for( unsigned i = 0 ; i < prod.size(); ++i ) {
    if(  prod[i].layer() != PFLayer::HGC_HCALB ) continue;
    const auto& pos = prod[i].position();
    const auto& pos_vect = pos - math::XYZPoint(0.0,0.0,0.0);
    auto x_rh = minmax(pos.x()+_hit_search_radius,pos.x()-_hit_search_radius);
    auto y_rh = minmax(pos.y()+_hit_search_radius,pos.y()-_hit_search_radius);
    auto z_rh = minmax(pos.z()+0.5*_hit_search_length,
		       pos.z()-0.5*_hit_search_length);
    KDTreeCube hit_searchcube((float)x_rh.first,(float)x_rh.second,
			      (float)y_rh.first,(float)y_rh.second,
			      (float)z_rh.first,(float)z_rh.second);
    _hit_kdtree.search(hit_searchcube,_found);
    double density_weight = 0.0;
    for( const KDNode& nbour : _found ) {
      if( nbour.data == i ) continue;
      const auto& npos = prod[nbour.data].position();
      const auto& diff = pos - npos;
      const double weight = pos_vect.Cross(diff).r()/pos_vect.r();
      density_weight += 1.0/std::pow(weight,_weight_power);
    }
    mask[i] = ( density_weight > std::pow(_weight_cut,_weight_power) );
    _found.clear();
  }
  // release what memory we can
  _found.clear();
  _hit_nodes.clear();
  _hit_kdtree.clear();
}

