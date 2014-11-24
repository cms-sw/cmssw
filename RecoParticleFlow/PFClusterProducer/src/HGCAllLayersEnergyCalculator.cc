#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

class HGCAllLayersEnergyCalibrator : public PFClusterEnergyCorrectorBase {
 public:
  HGCAllLayersEnergyCalibrator(const edm::ParameterSet& conf) :
    PFClusterEnergyCorrectorBase(conf),
    _mipValueInGeV_ee(conf.getParameter<double>("MipValueInGeV_ee")),
    _mipValueInGeV_hef(conf.getParameter<double>("MipValueInGeV_hef")),    
    _mipValueInGeV_heb(conf.getParameter<double>("MipValueInGeV_heb")),
    _coef_a(conf.getParameter<double>("effMip_to_InverseGeV_a")),
    _coef_b(conf.getParameter<double>("effMip_to_InverseGeV_b")),
    _coef_c(conf.getParameter<double>("effMip_to_InverseGeV_c")),     
    _weights_ee(conf.getParameter<std::vector<double> >("weights_ee")),
    _weights_hef(conf.getParameter<std::vector<double> >("weights_hef")),
    _weights_heb(conf.getParameter<std::vector<double> >("weights_heb"))
      { }
  HGCAllLayersEnergyCalibrator(const HGCAllLayersEnergyCalibrator&) = delete;
  HGCAllLayersEnergyCalibrator& operator=(const HGCAllLayersEnergyCalibrator&) = delete;
  
  void correctEnergy(reco::PFCluster& c) { correctEnergyActual(c); }
  void correctEnergies(reco::PFClusterCollection& cs) {
    for( unsigned i = 0; i < cs.size(); ++i ) correctEnergyActual(cs[i]);
  }

 private:  
  const double _mipValueInGeV_ee,_mipValueInGeV_hef,_mipValueInGeV_heb,
    _coef_a,_coef_b,_coef_c;
  const std::vector<double> _weights_ee,_weights_hef,_weights_heb;
  
  void correctEnergyActual(reco::PFCluster&) const;
  
};

DEFINE_EDM_PLUGIN(PFClusterEnergyCorrectorFactory,
		  HGCAllLayersEnergyCalibrator,
		  "HGCAllLayersEnergyCalibrator");

namespace {
  template<typename DETID>
  std::pair<int,int> getlayer(const unsigned rawid) {
    DETID id(rawid);
    return std::make_pair(id.zside(),id.layer());
  }
}

void HGCAllLayersEnergyCalibrator::
correctEnergyActual(reco::PFCluster& cluster) const {  
  //std::cout << "inside HGC EE correct energy" << std::endl;
  double eCorr = 0.0;
  const double clus_eta = cluster.positionREP().eta();
  const double effMIP_to_InvGeV = _coef_a/(1.0 + std::exp(-_coef_c - _coef_b*std::cosh(clus_eta)));  
  for( const auto& rhf : cluster.recHitFractions() ) {
    const std::vector<double>* weights = nullptr;
    const reco::PFRecHit& hit = *(rhf.recHitRef());
    DetId theid( hit.detId() );
    std::pair<int,int> zside_layer ;    
    double mip_value = 0.0;
    double mip2gev = 0.0;
    switch( theid.subdetId() ) {
    case HGCEE:
      weights = &_weights_ee;
      zside_layer = getlayer<HGCEEDetId>(hit.detId());
      mip_value = _mipValueInGeV_ee;
      mip2gev = effMIP_to_InvGeV;
      break;
    case HGCHEF:
      weights = &_weights_hef;
      zside_layer = getlayer<HGCHEDetId>(hit.detId());
      mip_value = _mipValueInGeV_hef;
      mip2gev = effMIP_to_InvGeV;
      break;
    case HGCHEB:
      weights = &_weights_heb;
      zside_layer = getlayer<HGCHEDetId>(hit.detId());
      mip_value = _mipValueInGeV_heb;
      mip2gev = effMIP_to_InvGeV;
      break;
    default:
      throw cms::Exception("BadRecHit")
	<< "This module only accepts HGC EE, HEF, or HEB hits" << std::endl;
    }
    const int layer = zside_layer.second;
    const double energy_MIP = hit.energy()/mip_value; 
    const double added_energy = (*weights)[layer-1]*energy_MIP/mip2gev;
    eCorr += added_energy;
  }  

  cluster.setEnergy(eCorr);
  cluster.setCorrectedEnergy(eCorr);
  
  //std::cout << "cluster pt: " << cluster.energy()/std::cosh(cluster.position().eta()) << std::endl;
}
