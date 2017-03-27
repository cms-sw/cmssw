#ifndef __RecoLocalCalo_HGCalRecAlgos_ClusterTools_h__
#define __RecoLocalCalo_HGCalRecAlgos_ClusterTools_h__

#include <array>
#include <cmath>

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"


class HGCalGeometry;
class HGCalDDDConstants;
class DetId;

namespace edm {
  class Event;
  class EventSetup;
}

namespace hgcal {
  class ClusterTools {    
  public:
    ClusterTools();
    ClusterTools(const edm::ParameterSet&, edm::ConsumesCollector&);
    ~ClusterTools() {}

    void getEvent(const edm::Event&);
    void getEventSetup(const edm::EventSetup&);        

    float getClusterHadronFraction(const reco::CaloCluster&) const;

    math::XYZPoint getMultiClusterPosition(const reco::HGCalMultiCluster&, double vz = 0.) const;
    
    int getLayer(const DetId) const;
    
    double getMultiClusterEnergy(const reco::HGCalMultiCluster&) const;

    // only for EE
    bool getWidths(const reco::CaloCluster & clus,double & sigmaetaeta, double & sigmaphiphi, double & sigmaetaetalog, double & sigmaphiphilog ) const;
  private:

    std::vector<size_t> sort_by_z(const reco::HGCalMultiCluster&v) const {
      std::vector<size_t> idx(v.size());
      for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
      sort(idx.begin(), idx.end(),
	   [&v](size_t i1, size_t i2) {return v.clusters()[i1]->z() < v.clusters()[i2]->z();});
      return idx;
    }

    RecHitTools rhtools_;
    const edm::EDGetTokenT<HGCRecHitCollection> eetok, fhtok, bhtok;
    const HGCRecHitCollection *eerh_, *fhrh_, *bhrh_;
  };
}

#endif
