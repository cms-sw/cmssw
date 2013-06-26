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
// hash function for edm::Ptr<reco::PFCluster
namespace std {
  template <> struct hash<edm::Ptr<reco::PFCluster> > {
    size_t operator()(const edm::Ptr<reco::PFCluster> & x) const {
      return hash<ptrdiff_t>()((ptrdiff_t)x.get());
    }
  };
}

class PFECALSuperClusterAlgo {  
 public:
  enum clustering_type{kBOX=1, kMustache=2};

  // simple class for associating calibrated energies
  class CalibratedPFCluster {
  public:
    CalibratedPFCluster(const edm::Ptr<reco::PFCluster>& p,
			const double ce) : cluptr(p), calib_e(ce) {}
    
    double energy() const { return calib_e; }
    double energy_nocalib() const { return cluptr->energy(); }
    double eta() const { return cluptr->positionREP().eta(); }
    double phi() const { return cluptr->positionREP().phi(); }
    
    void resetCalibratedEnergy(const double ce) { calib_e = ce; }
   
    edm::Ptr<reco::PFCluster> the_ptr() const { return cluptr; }

  private:
    edm::Ptr<reco::PFCluster> cluptr;
    double calib_e;
  };
  typedef std::shared_ptr<CalibratedPFCluster> CalibratedClusterPtr;
  typedef std::vector<CalibratedClusterPtr> CalibratedClusterPtrVector;


  /// constructor
  PFECALSuperClusterAlgo();

  void setVerbosityLevel(bool verbose){ verbose_ = verbose;}
  
  void setClusteringType(clustering_type thetype) { _clustype = thetype; } 

  void setUseDynamicDPhi(bool useit) { _useDynamicDPhi = useit; } 

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

  std::auto_ptr<reco::SuperClusterCollection>
    getEBOutputSCCollection() { return superClustersEB_; }
  std::auto_ptr<reco::SuperClusterCollection>
    getEEOutputSCCollection() { return superClustersEE_; }  

  void loadAndSortPFClusters(const edm::View<reco::PFCluster>& ecalclusters,
			     const edm::View<reco::PFCluster>& psclusters);
  
  void run();

 private:  

  CalibratedClusterPtrVector _clustersEB;
  CalibratedClusterPtrVector _clustersEE;
  std::unordered_map<edm::Ptr<reco::PFCluster>, 
    edm::PtrVector<reco::PFCluster> > _psclustersforee;
  std::auto_ptr<reco::SuperClusterCollection> superClustersEB_;
  std::auto_ptr<reco::SuperClusterCollection> superClustersEE_;
  std::shared_ptr<PFEnergyCalibration> _pfEnergyCalibration;
  clustering_type _clustype;
  void buildAllSuperClusters(CalibratedClusterPtrVector&,
			     double seedthresh);
  void buildSuperCluster(CalibratedClusterPtr&,
			 CalibratedClusterPtrVector&); 

  bool verbose_;

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
  
  bool usePS;

};

#endif
