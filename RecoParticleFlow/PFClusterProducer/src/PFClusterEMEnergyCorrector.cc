#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEMEnergyCorrector.h"

namespace {
  typedef reco::PFCluster::EEtoPSAssociation::value_type EEPSPair;
  bool sortByKey(const EEPSPair& a, const EEPSPair& b) {
    return a.first < b.first;
  } 
}

void PFClusterEMEnergyCorrector::
correctEnergyActual(reco::PFCluster& cluster, unsigned idx) const {
  std::vector<double> ps1_energies,ps2_energies;
  double ePS1=0, ePS2=0;
  if( cluster.layer() == PFLayer::ECAL_ENDCAP && _assoc ) {
    auto ee_key_val = std::make_pair(idx,edm::Ptr<reco::PFCluster>());
    const auto clustops = std::equal_range(_assoc->begin(),
					   _assoc->end(),
					   ee_key_val,
					   sortByKey);
    for( auto i_ps = clustops.first; i_ps != clustops.second; ++i_ps) {
      edm::Ptr<reco::PFCluster> psclus(i_ps->second);
      switch( psclus->layer() ) {
      case PFLayer::PS1:
	ps1_energies.push_back(psclus->energy());
	break;
      case PFLayer::PS2:
	ps2_energies.push_back(psclus->energy());
	break;
      default:
	break;
      }
    }
  }
  const double eCorr= _calibrator->energyEm(cluster,
					    ps1_energies,ps2_energies,
					    ePS1,ePS2,
					    _applyCrackCorrections);
  cluster.setCorrectedEnergy(eCorr);
}
