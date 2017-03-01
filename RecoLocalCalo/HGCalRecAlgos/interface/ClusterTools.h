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
    ClusterTools(const edm::ParameterSet&, edm::ConsumesCollector&);
    ~ClusterTools() {}

    void getEvent(const edm::Event&);
    void getEventSetup(const edm::EventSetup&);
    
    float getClusterHadronFraction(const reco::CaloCluster&) const;

    math::XYZPoint getMultiClusterPosition(const reco::HGCalMultiCluster&, double vz = 0.) const;
    
    
    double getMultiClusterEnergy(const reco::HGCalMultiCluster&) const;

  private:
    RecHitTools rhtools_;
    const edm::EDGetTokenT<HGCRecHitCollection> eetok, fhtok, bhtok;
    const HGCRecHitCollection *eerh_, *fhrh_, *bhrh_;
  };
}

#endif
