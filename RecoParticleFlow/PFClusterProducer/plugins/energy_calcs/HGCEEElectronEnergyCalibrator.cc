#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

class HGCEEElectronEnergyCalibrator : public PFClusterEnergyCorrectorBase {
 public:
  HGCEEElectronEnergyCalibrator(const edm::ParameterSet& conf) :
    PFClusterEnergyCorrectorBase(conf),
    _mipValueInGeV(conf.getParameter<double>("MipValueInGeV")),
    _coef_a(conf.getParameter<double>("effMip_to_InverseGeV_a")),
    _coef_b(conf.getParameter<double>("effMip_to_InverseGeV_b")),
    _coef_c(conf.getParameter<double>("effMip_to_InverseGeV_c")),
    _weights(conf.getParameter<std::vector<double> >("weights"))
      { }
  HGCEEElectronEnergyCalibrator(const HGCEEElectronEnergyCalibrator&) = delete;
  HGCEEElectronEnergyCalibrator& operator=(const HGCEEElectronEnergyCalibrator&) = delete;
  
  void correctEnergy(reco::PFCluster& c) { correctEnergyActual(c); }
  void correctEnergies(reco::PFClusterCollection& cs) {
    for( unsigned i = 0; i < cs.size(); ++i ) correctEnergyActual(cs[i]);
  }

 private:  
  const double _mipValueInGeV,_coef_a,_coef_b,_coef_c;
  const std::vector<double> _weights;
  
  void correctEnergyActual(reco::PFCluster&) const;
  
};

DEFINE_EDM_PLUGIN(PFClusterEnergyCorrectorFactory,
		  HGCEEElectronEnergyCalibrator,
		  "HGCEEElectronEnergyCalibrator");

namespace {
  template<typename DETID>
  std::pair<int,int> getlayer(const unsigned rawid) {
    DETID id(rawid);
    return std::make_pair(id.zside(),id.layer());
  }
}

void HGCEEElectronEnergyCalibrator::
correctEnergyActual(reco::PFCluster& cluster) const {  
  //std::cout << "inside HGC EE correct energy" << std::endl;
  double eCorr = 0.0;
  const double clus_eta = cluster.positionREP().eta();
  for( const auto& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHit& hit = *(rhf.recHitRef());    
    DetId id(hit.detId());
    if( id.det() != DetId::Forward ) continue;    
    switch( id.subdetId() ) {
    case HGCEE:    
      {
	const std::pair<int,int> zside_layer = 
	  getlayer<HGCEEDetId>(hit.detId());
	const int layer = zside_layer.second;
	//const double eta = hit.positionREP().eta();
	const double energy_MIP = hit.energy()/_mipValueInGeV;
	eCorr += _weights[layer-1]*energy_MIP;//std::cosh(eta);
      }
      break;
    default:
      break;
    }
  }
  
  /*
  if( cluster.recHitFractions().size() > 50 ) {
    std::cout << " --------- " 
	      << cluster.position().eta() << ' ' 
	      << cluster.positionREP().eta() << ' '
	      << eCorr << ' ' 
	      << cluster.recHitFractions().size() << std::endl;
  }
  */

  const double effMIP_to_InvGeV = _coef_a/(1.0 + std::exp(-_coef_c - _coef_b*std::cosh(clus_eta)));

  cluster.setEnergy(eCorr/effMIP_to_InvGeV);
  cluster.setCorrectedEnergy(eCorr/effMIP_to_InvGeV);
}
