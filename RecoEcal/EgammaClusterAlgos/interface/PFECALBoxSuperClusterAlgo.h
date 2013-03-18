#ifndef RecoEcal_EgammaClusterAlgos_PFECALBoxSuperClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_PFECALBoxSuperClusterAlgo_h


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

#include <string>
#include <vector>
#include <map>
#include <set>

#include <memory>

class TFile;
class TH2F;

/// \ Algorithm for box particle flow super clustering in the ECAL
/*!

  \author Nicolas Chanon
  \date July 2012
*/

struct less_magPF : public std::binary_function<reco::PFClusterRef, reco::PFClusterRef, bool> {
  bool operator()(reco::PFClusterRef x, reco::PFClusterRef y) { return x->energy() > y->energy() ; }
};


class PFECALBoxSuperClusterAlgo {

 public:

  /// constructor
  PFECALBoxSuperClusterAlgo();

  /// destructor
  virtual ~PFECALBoxSuperClusterAlgo() {;}

  void setVerbosityLevel(bool verbose){ verbose_ = verbose;}

  void setThreshPFClusterSeedBarrel(double thresh){ threshPFClusterSeedBarrel_ = thresh;}
  void setThreshPFClusterBarrel(double thresh){ threshPFClusterBarrel_ = thresh;}
  void setThreshPFClusterSeedEndcap(double thresh){ threshPFClusterSeedEndcap_ = thresh;}
  void setThreshPFClusterEndcap(double thresh){ threshPFClusterEndcap_ = thresh;}
  
  void setPhiwidthSuperClusterBarrel( double phiwidth ){ phiwidthSuperClusterBarrel_ = phiwidth;}
  void setEtawidthSuperClusterBarrel( double etawidth ){ etawidthSuperClusterBarrel_ = etawidth;}
  void setPhiwidthSuperClusterEndcap( double phiwidth ){ phiwidthSuperClusterEndcap_ = phiwidth;}
  void setEtawidthSuperClusterEndcap( double etawidth ){ etawidthSuperClusterEndcap_ = etawidth;}

  void setThreshPFClusterES(double thresh){threshPFClusterES_ = thresh;}
  
  void setMustacheCut( bool doMustacheCut ) { doMustacheCut_ = doMustacheCut;}
  //void setThreshPFClusterMustacheOutBarrel(double thresh){ threshPFClusterMustacheOutBarrel_ = thresh;}
  //void setThreshPFClusterMustacheOutEndcap(double thresh){ threshPFClusterMustacheOutEndcap_ = thresh;}

  void setCrackCorrections( bool applyCrackCorrections) { applyCrackCorrections_ = applyCrackCorrections;}

  void doClustering(const edm::Handle<reco::PFClusterCollection> & pfclustersHandle, std::auto_ptr< reco::BasicClusterCollection > & basicClusters_p, boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_, int detector);

  void matchSCtoESclusters(const edm::Handle<reco::PFClusterCollection> & pfclustersHandl, std::auto_ptr< reco::SuperClusterCollection > & pfSuperClustersWithES_p, boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_, int detector);

  void findClustersOutsideMustacheArea();

  void storeSuperClusters(const edm::OrphanHandle<reco::BasicClusterCollection> & basicClustersHandle, std::auto_ptr< reco::SuperClusterCollection > & pfSuperClusters_p );


 private:

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

  bool doMustacheCut_;
  //double threshPFClusterMustacheOutBarrel_;
  //double threshPFClusterMustacheOutEndcap_;

  bool applyCrackCorrections_;

  int nSuperClusters;

  std::vector<int> scPFseedIndex_;
  std::vector<int> seedCandidateIndex_;
  std::vector<int> pfClusterIndex_;

  std::vector<std::vector<const reco::PFCluster *> > pfClusters_;
  std::vector< reco::BasicClusterCollection > basicClusters_;

  std::vector<reco::CaloClusterPtrVector> basicClusterPtr_;

  std::vector<double> allPfClusterCalibratedEnergy_;
  std::vector<std::vector<double>> pfClusterCalibratedEnergy_;
  std::vector<std::vector<double>> pfClusterCalibratedEnergyWithES_;

  std::vector<reco::PFClusterRef> seedCandidateCollection;
  std::vector<reco::PFClusterRef> pfClusterAboveThresholdCollection;
  //  std::vector<reco::PFClusterRef> pfESClusterAboveThresholdCollection;

  //  std::vector<double>** SCBCtoESenergyPS1;
  //  std::vector<double>** SCBCtoESenergyPS2;

  //  std::vector<int> isSeedUsed;
  //  std::vector<int> isPFclusterUsed;
  //  std::vector<bool> isClusterized;

  std::vector<std::vector<unsigned int>> insideMust_;
  //std::vector<std::vector<unsigned int>> outsideMust_;


  void createBasicCluster(const reco::PFClusterRef & myPFClusterRef, 
					      reco::BasicClusterCollection & basicClusters, 
					      std::vector<const reco::PFCluster *> & pfClusters) const;
  
  void createBasicClusterPtrs(const edm::OrphanHandle<reco::BasicClusterCollection> & basicClustersHandle ) ;

  void createSuperClusters(reco::SuperClusterCollection &superClusters, bool doEEwithES) const;

  reco::SuperClusterCollection superClusters_;

};

#endif
