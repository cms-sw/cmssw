#ifndef RecoParticleFlow_PFClusterTools_LinkByRecHit_h
#define RecoParticleFlow_PFClusterTools_LinkByRecHit_h 

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"



class LinkByRecHit {
 public:
  LinkByRecHit() {};
  ~LinkByRecHit() {} ; 

  /// computes a chisquare
  static double computeDist( double eta1, double phi1, 
			     double eta2, double phi2,
			     bool etaPhi = true) ;
  
  //tests association between a track and a cluster by rechit
  static double testTrackAndClusterByRecHit( const reco::PFRecTrack& track, 
					     const reco::PFCluster& cluster,
					     bool isBrem = false,
					     bool debug = false ) ;  

  //tests association between ECAL and PS clusters by rechit
  static double testECALAndPSByRecHit( const reco::PFCluster& clusterECAL, 
				       const reco::PFCluster& clusterPS,
				       bool debug=false)  ;
  
  /// test association between HFEM and HFHAD, by rechit
  static double testHFEMAndHFHADByRecHit( const reco::PFCluster& clusterHFEM, 
					  const reco::PFCluster& clusterHFHAD,
					  bool debug=false)  ;
  

};

#endif
