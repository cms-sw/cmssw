#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEcal/EgammaClusterAlgos/interface/Multi5x5ClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <ctime>
#include <iostream>
#include <memory>
#include <vector>

class Multi5x5ClusterProducer : public edm::stream::EDProducer<> {
public:
  Multi5x5ClusterProducer(const edm::ParameterSet& ps);

  void produce(edm::Event&, const edm::EventSetup&) final;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<EcalRecHitCollection> barrelHitToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> endcapHitToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;

  const edm::EDPutTokenT<reco::BasicClusterCollection> barrelToken_;
  const edm::EDPutTokenT<reco::BasicClusterCollection> endcapToken_;

  Multi5x5ClusterAlgo island_;

  // cluster which regions
  const bool doBarrel_;
  const bool doEndcap_;

  reco::BasicClusterCollection clusterizeECALPart(const EcalRecHitCollection& hits,
                                                  const edm::EventSetup& es,
                                                  const reco::CaloID::Detectors detector);

  reco::BasicClusterCollection makeClusters(const EcalRecHitCollection& hits,
                                            const CaloSubdetectorGeometry* geom,
                                            const CaloSubdetectorGeometry* preshower,
                                            const CaloSubdetectorTopology* topology,
                                            const reco::CaloID::Detectors detector);
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Multi5x5ClusterProducer);

Multi5x5ClusterProducer::Multi5x5ClusterProducer(const edm::ParameterSet& ps)
    : barrelHitToken_{consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("barrelHitTag"))},
      endcapHitToken_{consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("endcapHitTag"))},
      caloGeometryToken_{esConsumes<CaloGeometry, CaloGeometryRecord>()},
      barrelToken_{produces<reco::BasicClusterCollection>(ps.getParameter<std::string>("barrelClusterCollection"))},
      endcapToken_{produces<reco::BasicClusterCollection>(ps.getParameter<std::string>("endcapClusterCollection"))},
      island_{ps.getParameter<double>("IslandBarrelSeedThr"),
              ps.getParameter<double>("IslandEndcapSeedThr"),
              StringToEnumValue<EcalRecHit::Flags>(ps.getParameter<std::vector<std::string>>("RecHitFlagToBeExcluded")),
              PositionCalc(ps.getParameter<edm::ParameterSet>("posCalcParameters")),
              ps.getParameter<bool>("reassignSeedCrysToClusterItSeeds")},
      doBarrel_{ps.getParameter<bool>("doBarrel")},
      doEndcap_{ps.getParameter<bool>("doEndcap")} {}

void Multi5x5ClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
  edm::ParameterSetDescription ps;
  ps.add<edm::InputTag>("barrelHitTag");
  ps.add<edm::InputTag>("endcapHitTag");
  ps.add<bool>("doEndcap");
  ps.add<bool>("doBarrel");
  ps.add<std::string>("barrelClusterCollection");
  ps.add<std::string>("endcapClusterCollection");
  ps.add<double>("IslandBarrelSeedThr");
  ps.add<double>("IslandEndcapSeedThr");
  ps.add<std::vector<std::string>>("RecHitFlagToBeExcluded");

  edm::ParameterSetDescription posCal;
  posCal.add<bool>("LogWeighted");
  posCal.add<double>("T0_barl");
  posCal.add<double>("T0_endc");
  posCal.add<double>("T0_endcPresh");
  posCal.add<double>("W0");
  posCal.add<double>("X0");
  ps.add<edm::ParameterSetDescription>("posCalcParameters", posCal);

  ps.add<bool>("reassignSeedCrysToClusterItSeeds", false);
  iDesc.addDefault(ps);
}

void Multi5x5ClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  if (doEndcap_) {
    // get the hit collection from the event:
    const EcalRecHitCollection& hitCollection = evt.get(endcapHitToken_);
    evt.emplace(endcapToken_, clusterizeECALPart(hitCollection, es, reco::CaloID::DET_ECAL_ENDCAP));
  }
  if (doBarrel_) {
    // get the hit collection from the event:
    const EcalRecHitCollection& hitCollection = evt.get(barrelHitToken_);
    evt.emplace(barrelToken_, clusterizeECALPart(hitCollection, es, reco::CaloID::DET_ECAL_BARREL));
  }
}

reco::BasicClusterCollection Multi5x5ClusterProducer::makeClusters(const EcalRecHitCollection& hits,
                                                                   const CaloSubdetectorGeometry* geom,
                                                                   const CaloSubdetectorGeometry* preshower,
                                                                   const CaloSubdetectorTopology* topology,
                                                                   const reco::CaloID::Detectors detector) {
  // Run the clusterization algorithm:
  return island_.makeClusters(&hits, geom, topology, preshower, detector);
}

reco::BasicClusterCollection Multi5x5ClusterProducer::clusterizeECALPart(const EcalRecHitCollection& hitCollection,
                                                                         const edm::EventSetup& es,
                                                                         const reco::CaloID::Detectors detector) {
  // get the geometry and topology from the event setup:
  CaloGeometry const& geo = es.getData(caloGeometryToken_);

  const CaloSubdetectorGeometry* preshower_p = geo.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  if (detector == reco::CaloID::DET_ECAL_BARREL) {
    auto geometry_p = geo.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    EcalBarrelTopology topology{geo};
    return makeClusters(hitCollection, geometry_p, preshower_p, &topology, detector);
  } else {
    auto geometry_p = geo.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    EcalEndcapTopology topology{geo};
    return makeClusters(hitCollection, geometry_p, preshower_p, &topology, detector);
  }
}
