#ifndef RecoEcal_EgammaClusterAlgos_PFECALSuperClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_PFECALSuperClusterAlgo_h

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

#include "RecoEcal/EgammaClusterAlgos/interface/SCEnergyCorrectorSemiParm.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClustersGraph.h"
#include "RecoEcal/EgammaCoreTools/interface/CalibratedPFCluster.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/DataRecord/interface/ESEEIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/ESChannelStatusRcd.h"
#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"

#include "CondFormats/EcalObjects/interface/EcalMustacheSCParameters.h"
#include "CondFormats/DataRecord/interface/EcalMustacheSCParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalSCDynamicDPhiParameters.h"
#include "CondFormats/DataRecord/interface/EcalSCDynamicDPhiParametersRcd.h"

#include "RecoEcal/EgammaCoreTools/interface/SCProducerCache.h"

#include <vector>
#include <memory>

/// \ Algorithm for box particle flow super clustering in the ECAL
/*!

  Original Author: Nicolas Chanon
  Additional Authors (Mustache): Y. Gershtein, R. Patel, L. Gray
  \date July 2012
*/

typedef std::vector<CalibratedPFCluster> CalibratedPFClusterVector;

class PFECALSuperClusterAlgo {
public:
  enum clustering_type { kBOX = 1, kMustache = 2, kDeepSC = 3 };
  enum energy_weight { kRaw, kCalibratedNoPS, kCalibratedTotal };

  /// constructor
  PFECALSuperClusterAlgo(const reco::SCProducerCache* cache);

  void setVerbosityLevel(bool verbose) { verbose_ = verbose; }

  void setClusteringType(clustering_type thetype) { _clustype = thetype; }

  void setEnergyWeighting(energy_weight thetype) { _eweight = thetype; }

  void setUseETForSeeding(bool useET) { threshIsET_ = useET; }

  void setUseDynamicDPhi(bool useit) { useDynamicDPhi_ = useit; }

  void setUseRegression(bool useRegression) { useRegression_ = useRegression; }

  void setThreshSuperClusterEt(double thresh) { threshSuperClusterEt_ = thresh; }

  void setThreshPFClusterSeedBarrel(double thresh) { threshPFClusterSeedBarrel_ = thresh; }
  void setThreshPFClusterBarrel(double thresh) { threshPFClusterBarrel_ = thresh; }
  void setThreshPFClusterSeedEndcap(double thresh) { threshPFClusterSeedEndcap_ = thresh; }
  void setThreshPFClusterEndcap(double thresh) { threshPFClusterEndcap_ = thresh; }

  void setPhiwidthSuperClusterBarrel(double phiwidth) { phiwidthSuperClusterBarrel_ = phiwidth; }
  void setEtawidthSuperClusterBarrel(double etawidth) { etawidthSuperClusterBarrel_ = etawidth; }
  void setPhiwidthSuperClusterEndcap(double phiwidth) { phiwidthSuperClusterEndcap_ = phiwidth; }
  void setEtawidthSuperClusterEndcap(double etawidth) { etawidthSuperClusterEndcap_ = etawidth; }

  void setPFClusterCalibration(const std::shared_ptr<PFEnergyCalibration>&);

  void setSatelliteMerging(const bool doit) { doSatelliteClusterMerge_ = doit; }
  void setSatelliteThreshold(const double t) { satelliteThreshold_ = t; }
  void setMajorityFraction(const double f) { fractionForMajority_ = f; }
  void setDropUnseedable(const bool d) { dropUnseedable_ = d; }

  void setIsOOTCollection(bool isOOTCollection) { isOOTCollection_ = isOOTCollection; }

  void setCrackCorrections(bool applyCrackCorrections) { applyCrackCorrections_ = applyCrackCorrections; }

  void setTokens(const edm::ParameterSet&, edm::ConsumesCollector&&);

  void update(const edm::EventSetup&);
  void updateSCParams(const edm::EventSetup&);

  std::unique_ptr<reco::SuperClusterCollection>& getEBOutputSCCollection() { return superClustersEB_; }
  std::unique_ptr<reco::SuperClusterCollection>& getEEOutputSCCollection() { return superClustersEE_; }

  void loadAndSortPFClusters(const edm::Event& evt);

  void run();

private:
  edm::EDGetTokenT<edm::View<reco::PFCluster> > inputTagPFClusters_;
  edm::EDGetTokenT<reco::PFCluster::EEtoPSAssociation> inputTagPFClustersES_;
  edm::EDGetTokenT<reco::BeamSpot> inputTagBeamSpot_;

  edm::ESGetToken<ESEEIntercalibConstants, ESEEIntercalibConstantsRcd> esEEInterCalibToken_;
  edm::ESGetToken<ESChannelStatus, ESChannelStatusRcd> esChannelStatusToken_;
  edm::ESGetToken<EcalMustacheSCParameters, EcalMustacheSCParametersRcd> ecalMustacheSCParametersToken_;
  edm::ESGetToken<EcalSCDynamicDPhiParameters, EcalSCDynamicDPhiParametersRcd> ecalSCDynamicDPhiParametersToken_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopologyToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;

  const reco::BeamSpot* beamSpot_;
  const ESChannelStatus* channelStatus_;
  const CaloGeometry* geometry_;
  const CaloSubdetectorGeometry* ebGeom_;
  const CaloSubdetectorGeometry* eeGeom_;
  const CaloSubdetectorGeometry* esGeom_;
  const CaloTopology* topology_;

  const EcalMustacheSCParameters* mustacheSCParams_;
  const EcalSCDynamicDPhiParameters* scDynamicDPhiParams_;

  CalibratedPFClusterVector _clustersEB;
  CalibratedPFClusterVector _clustersEE;
  std::unique_ptr<reco::SuperClusterCollection> superClustersEB_;
  std::unique_ptr<reco::SuperClusterCollection> superClustersEE_;
  const reco::PFCluster::EEtoPSAssociation* EEtoPS_;
  std::shared_ptr<PFEnergyCalibration> _pfEnergyCalibration;
  clustering_type _clustype;
  energy_weight _eweight;
  void buildAllSuperClusters(CalibratedPFClusterVector&, double seedthresh);
  void buildAllSuperClustersMustacheOrBox(CalibratedPFClusterVector&, double seedthresh);
  void buildAllSuperClustersDeepSC(CalibratedPFClusterVector&, double seedthresh);
  void buildSuperClusterMustacheOrBox(CalibratedPFCluster&, CalibratedPFClusterVector&);
  void finalizeSuperCluster(CalibratedPFCluster& seed, CalibratedPFClusterVector& clustered, bool isEE);

  bool verbose_;

  // regression
  bool useRegression_;
  std::unique_ptr<SCEnergyCorrectorSemiParm> regr_;

  double threshSuperClusterEt_;

  double threshPFClusterSeedBarrel_;
  double threshPFClusterBarrel_;
  double threshPFClusterSeedEndcap_;
  double threshPFClusterEndcap_;

  double phiwidthSuperClusterBarrel_;
  double etawidthSuperClusterBarrel_;
  double phiwidthSuperClusterEndcap_;
  double etawidthSuperClusterEndcap_;

  bool doSatelliteClusterMerge_;  //rock it
  double satelliteThreshold_, fractionForMajority_;
  bool dropUnseedable_;

  bool useDynamicDPhi_;

  bool applyCrackCorrections_;
  bool threshIsET_;

  const reco::SCProducerCache* SCProducerCache_;

  // OOT photons
  bool isOOTCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> inputTagBarrelRecHits_;
  edm::EDGetTokenT<EcalRecHitCollection> inputTagEndcapRecHits_;
  const EcalRecHitCollection* barrelRecHits_;
  const EcalRecHitCollection* endcapRecHits_;
};

#endif
