#ifndef RecoEcal_EgammaClusterAlgos_PFECALSuperClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_PFECALSuperClusterAlgo_h


#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoEgamma/EgammaTools/interface/BaselinePFSCRegression.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "TVector2.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <set>

#include <memory>

class TFile;
class TH2F;

/// \ Algorithm for box particle flow super clustering in the ECAL
/*!

  Original Author: Nicolas Chanon
  Additional Authors (Mustache): Y. Gershtein, R. Patel, L. Gray
  \date July 2012
*/

class PFECALSuperClusterAlgo {  
 public:
  enum clustering_type{kBOX=1, kMustache=2};
  enum energy_weight{kRaw, kCalibratedNoPS, kCalibratedTotal};

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

  void setVerbosityLevel(bool verbose){ verbose_ = verbose;}
  
  void setClusteringType(clustering_type thetype) { _clustype = thetype; } 

  void setEnergyWeighting(energy_weight thetype) { _eweight = thetype; } 

  void setUseETForSeeding(bool useET) { threshIsET_ = useET; } 

  void setUseDynamicDPhi(bool useit) { _useDynamicDPhi = useit; } 

  void setUseRegression(bool useRegression) { useRegression_ = useRegression; }
  
  void setThreshSuperClusterEt(double thresh) { threshSuperClusterEt_ = thresh; }
  
  void setThreshPFClusterSeedBarrel(double thresh){ threshPFClusterSeedBarrel_ = thresh;}
  void setThreshPFClusterBarrel(double thresh){ threshPFClusterBarrel_ = thresh;}
  void setThreshPFClusterSeedEndcap(double thresh){ threshPFClusterSeedEndcap_ = thresh;}
  void setThreshPFClusterEndcap(double thresh){ threshPFClusterEndcap_ = thresh;}
  
  void setPhiwidthSuperClusterBarrel( double phiwidth ){ phiwidthSuperClusterBarrel_ = phiwidth;}
  void setEtawidthSuperClusterBarrel( double etawidth ){ etawidthSuperClusterBarrel_ = etawidth;}
  void setPhiwidthSuperClusterEndcap( double phiwidth ){ phiwidthSuperClusterEndcap_ = phiwidth;}
  void setEtawidthSuperClusterEndcap( double etawidth ){ etawidthSuperClusterEndcap_ = etawidth;}
  void setUsePS( bool useit ){ usePS = useit; }

  void setPFClusterCalibration(const std::shared_ptr<PFEnergyCalibration>&);
  
  void setThreshPFClusterES(double thresh){threshPFClusterES_ = thresh;}
  
  void setSatelliteMerging( const bool doit ) { doSatelliteClusterMerge_ = doit; }
  void setSatelliteThreshold( const double t ) { satelliteThreshold_ = t; }
  void setMajorityFraction( const double f ) { fractionForMajority_ = f; }
  //void setThreshPFClusterMustacheOutBarrel(double thresh){ threshPFClusterMustacheOutBarrel_ = thresh;}
  //void setThreshPFClusterMustacheOutEndcap(double thresh){ threshPFClusterMustacheOutEndcap_ = thresh;}

  void setCrackCorrections( bool applyCrackCorrections) { applyCrackCorrections_ = applyCrackCorrections;}
  
  void setTokens(const edm::ParameterSet&, edm::ConsumesCollector&&);
  void update(const edm::EventSetup&);
  
  
  std::auto_ptr<reco::SuperClusterCollection>&
    getEBOutputSCCollection() { return superClustersEB_; }
  std::auto_ptr<reco::SuperClusterCollection>&
    getEEOutputSCCollection() { return superClustersEE_; }

  void loadAndSortPFClusters(const edm::Event &evt);
  
  void run();

 private:  

  edm::EDGetTokenT<edm::View<reco::PFCluster> >   inputTagPFClusters_;
  edm::EDGetTokenT<reco::PFCluster::EEtoPSAssociation>   inputTagPFClustersES_;   
  edm::EDGetTokenT<reco::BeamSpot>   inputTagBeamSpot_;
   
  const reco::BeamSpot *beamSpot_;
  
  CalibratedClusterPtrVector _clustersEB;
  CalibratedClusterPtrVector _clustersEE;
  std::auto_ptr<reco::SuperClusterCollection> superClustersEB_;
  std::auto_ptr<reco::SuperClusterCollection> superClustersEE_;
  const reco::PFCluster::EEtoPSAssociation* EEtoPS_;
  std::shared_ptr<PFEnergyCalibration> _pfEnergyCalibration;
  clustering_type _clustype;
  energy_weight   _eweight;
  void buildAllSuperClusters(CalibratedClusterPtrVector&,
			     double seedthresh);
  void buildSuperCluster(CalibratedClusterPtr&,
			 CalibratedClusterPtrVector&); 

  bool verbose_;
  
  // regression
  bool useRegression_;
  std::unique_ptr<PFSCRegressionCalc> regr_;  
  
  double threshSuperClusterEt_;  

  double threshPFClusterSeed_;
  double threshPFCluster_;
  double etawidthSuperCluster_;
  double phiwidthSuperCluster_;

  double threshPFClusterSeedBarrel_;
  double threshPFClusterBarrel_;
  double threshPFClusterSeedEndcap_;
  double threshPFClusterEndcap_;
  double threshPFClusterES_;

  double phiwidthSuperClusterBarrel_;
  double etawidthSuperClusterBarrel_;
  double phiwidthSuperClusterEndcap_;
  double etawidthSuperClusterEndcap_;

  bool doSatelliteClusterMerge_; //rock it
  double satelliteThreshold_, fractionForMajority_;

  bool _useDynamicDPhi;

  bool applyCrackCorrections_;
  bool threshIsET_;

  bool usePS;

};

#endif
