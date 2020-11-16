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

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

#include "RecoEcal/EgammaClusterAlgos/interface/SCEnergyCorrectorSemiParm.h"

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

#include <vector>
#include <memory>

/// \ Algorithm for box particle flow super clustering in the ECAL
/*!

  Original Author: Nicolas Chanon
  Additional Authors (Mustache): Y. Gershtein, R. Patel, L. Gray
  \date July 2012
*/

class PFECALSuperClusterAlgo {
public:
  enum clustering_type { kBOX = 1, kMustache = 2 };
  enum energy_weight { kRaw, kCalibratedNoPS, kCalibratedTotal };

  // simple class for associating calibrated energies
  class CalibratedPFCluster {
  public:
    CalibratedPFCluster(const edm::Ptr<reco::PFCluster>& p) : cluptr(p) {}

    double energy() const { return cluptr->correctedEnergy(); }
    double energy_nocalib() const { return cluptr->energy(); }
    double eta() const { return cluptr->positionREP().eta(); }
    double phi() const { return cluptr->positionREP().phi(); }

    edm::Ptr<reco::PFCluster> the_ptr() const { return cluptr; }

  private:
    edm::Ptr<reco::PFCluster> cluptr;
  };
  typedef std::shared_ptr<CalibratedPFCluster> CalibratedClusterPtr;
  typedef std::vector<CalibratedClusterPtr> CalibratedClusterPtrVector;

  /// constructor
  PFECALSuperClusterAlgo();

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

  const reco::BeamSpot* beamSpot_;
  const ESChannelStatus* channelStatus_;
  const EcalMustacheSCParameters* mustacheSCParams_;
  const EcalSCDynamicDPhiParameters* scDynamicDPhiParams_;

  CalibratedClusterPtrVector _clustersEB;
  CalibratedClusterPtrVector _clustersEE;
  std::unique_ptr<reco::SuperClusterCollection> superClustersEB_;
  std::unique_ptr<reco::SuperClusterCollection> superClustersEE_;
  const reco::PFCluster::EEtoPSAssociation* EEtoPS_;
  std::shared_ptr<PFEnergyCalibration> _pfEnergyCalibration;
  clustering_type _clustype;
  energy_weight _eweight;
  void buildAllSuperClusters(CalibratedClusterPtrVector&, double seedthresh);
  void buildSuperCluster(CalibratedClusterPtr&, CalibratedClusterPtrVector&);

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

  // OOT photons
  bool isOOTCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> inputTagBarrelRecHits_;
  edm::EDGetTokenT<EcalRecHitCollection> inputTagEndcapRecHits_;
  const EcalRecHitCollection* barrelRecHits_;
  const EcalRecHitCollection* endcapRecHits_;
};

#endif
