#ifndef RecoParticleFlow_PFClusterProducer_h_
#define RecoParticleFlow_PFClusterProducer_h_

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

// #include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

// -*- C++ -*-
//
// Package:    PFClusterProducer
// Class:      PFClusterProducer
// 
/**\class PFClusterProducer PFClusterProducer.cc RecoParticleFlow/PFClusterProducer/src/PFClusterProducer.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Colin Bernet
//         Created:  Tue Jun 27 14:35:24 CEST 2006
// $Id: PFClusterProducer.h,v 1.1 2006/07/21 16:22:51 cbern Exp $
//
//

class CaloSubdetectorTopology;
class CaloSubdetectorGeometry;
class DetId;

class PFClusterProducer : public edm::EDProducer {
 public:
  explicit PFClusterProducer(const edm::ParameterSet&);
  ~PFClusterProducer();

  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
  /// find and set the neighbours to a given rechit
  void FindRecHitNeighbours( reco::PFRecHit* rh, 
			     const std::map<unsigned,  reco::PFRecHit* >& rechits, 
			     const CaloSubdetectorTopology& barrelTopology,
			     const CaloSubdetectorGeometry& barrelGeometry, 
			     const CaloSubdetectorTopology& endcapTopology,
			     const CaloSubdetectorGeometry& endcapGeometry );
 
/* 			     const CaloSubdetectorTopology& hcalTopology, */
/* 			     const CaloSubdetectorGeometry& hcalBarrelGeometry, */
/* 			     const CaloSubdetectorGeometry& hcalEndcapGeometry ); */
  
  reco::PFRecHit*  CreateHcalRecHit( const DetId& detid, 
				     double energy,
				     int layer,
				     const CaloSubdetectorGeometry* geom );

 private:
  // ----------member data ---------------------------

  /// process Ecal ? 
  bool   processEcal_;

  /// process Hcal ? 
  bool   processHcal_;  

  /// process preshower ? 
  bool   processPS_;  

  /// ecal barrel threshold
  double threshEcalBarrel_;

  /// ecal barrel seed threshold
  double threshSeedEcalBarrel_;

  /// ecal endcap threshold
  double threshEcalEndcap_;

  /// ecal endcap seed threshold
  double threshSeedEcalEndcap_;



  /// ps threshold
  double threshPS_;

  /// ps seed threshold
  double threshSeedPS_;



  /// hcal barrel threshold
  double threshHcalBarrel_;

  /// hcal barrel seed threshold
  double threshSeedHcalBarrel_;

  /// hcal endcap threshold
  double threshHcalEndcap_;

  /// hcal endcap seed threshold
  double threshSeedHcalEndcap_;

};

#endif
