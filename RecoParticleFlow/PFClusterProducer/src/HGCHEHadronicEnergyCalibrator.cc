#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

class HGCHEHadronicEnergyCalibrator : public PFClusterEnergyCorrectorBase {
 public:
  HGCHEHadronicEnergyCalibrator(const edm::ParameterSet& conf) :
    PFClusterEnergyCorrectorBase(conf),
    _mipValueInGeV_HEF(conf.getParameter<double>("MipValueInGeV_HEF")),    
    _coef_a_hef(conf.getParameter<double>("effMip_to_InverseGeV_a_HEF")),
    _coef_b_hef(conf.getParameter<double>("effMip_to_InverseGeV_b_HEF")),
    _coef_c_hef(conf.getParameter<double>("effMip_to_InverseGeV_c_HEF")),
    _mipValueInGeV_HEB(conf.getParameter<double>("MipValueInGeV_HEB")),
    _coef_a_heb(conf.getParameter<double>("effMip_to_InverseGeV_a_HEB")),
    _coef_b_heb(conf.getParameter<double>("effMip_to_InverseGeV_b_HEB")),
    _coef_c_heb(conf.getParameter<double>("effMip_to_InverseGeV_c_HEB")),
    _weights_hef(conf.getParameter<std::vector<double> >("weights_hef")),
    _weights_heb(conf.getParameter<std::vector<double> >("weights_heb"))
      { }
  HGCHEHadronicEnergyCalibrator(const HGCHEHadronicEnergyCalibrator&) = delete;
  HGCHEHadronicEnergyCalibrator& operator=(const HGCHEHadronicEnergyCalibrator&) = delete;
  
  void correctEnergy(reco::PFCluster& c) { correctEnergyActual(c); }
  void correctEnergies(reco::PFClusterCollection& cs) {
    for( unsigned i = 0; i < cs.size(); ++i ) correctEnergyActual(cs[i]);
  }

 private:  
  const double _mipValueInGeV_HEF,_coef_a_hef,_coef_b_hef,_coef_c_hef;
  const double _mipValueInGeV_HEB,_coef_a_heb,_coef_b_heb,_coef_c_heb;
  const std::vector<double> _weights_hef,_weights_heb;
  
  void correctEnergyActual(reco::PFCluster&) const;
  
};

DEFINE_EDM_PLUGIN(PFClusterEnergyCorrectorFactory,
		  HGCHEHadronicEnergyCalibrator,
		  "HGCHEHadronicEnergyCalibrator");

namespace {
  template<typename DETID>
  std::pair<int,int> getlayer(const unsigned rawid) {
    DETID id(rawid);
    return std::make_pair(id.zside(),id.layer());
  }
}

void HGCHEHadronicEnergyCalibrator::
correctEnergyActual(reco::PFCluster& cluster) const {  
  //std::cout << "inside HGC EE correct energy" << std::endl;
  double eCorr = 0.0;
  const double clus_eta = cluster.positionREP().eta();
  const double effMIP_to_InvGeV_HEF = _coef_a_hef/(1.0 + std::exp(-_coef_c_hef - _coef_b_hef*std::cosh(clus_eta)));
  const double effMIP_to_InvGeV_HEB = _coef_a_heb/(1.0 + std::exp(-_coef_c_heb - _coef_b_heb*std::cosh(clus_eta)));
  for( const auto& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHit& hit = *(rhf.recHitRef());
    HGCHEDetId theid( hit.detId() );
    const std::pair<int,int> zside_layer = getlayer<HGCHEDetId>(hit.detId());
    const int layer = zside_layer.second;
    double mip_value = 0.0;
    double mip2gev = 0.0;
    double weight = 0.0;
    switch( theid.subdet() ) {
    case HGCHEF:
      mip_value = _mipValueInGeV_HEF;
      mip2gev = effMIP_to_InvGeV_HEF;
      weight = _weights_hef[layer-1];
      break;
    case HGCHEB:
      mip_value = _mipValueInGeV_HEB;
      mip2gev = effMIP_to_InvGeV_HEB;
      weight = _weights_heb[layer-1];
      break;
    default:
      throw cms::Exception("BadRecHit")
	<< "This module only accepts HGC HEF or HEB hits" << std::endl;
    }
    const double energy_MIP = hit.energy()/mip_value; 
    eCorr += weight*energy_MIP/mip2gev;
  }  

  cluster.setEnergy(eCorr);
  cluster.setCorrectedEnergy(eCorr);
}
