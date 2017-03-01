#ifndef RECOMET_METALGORITHMS_HCALHALOALGO_H
#define RECOMET_METALGORITHMS_HCALHALOALGO_H

#include "DataFormats/METReco/interface/HcalHaloData.h"

/*
  [class]:  HcalHaloAlgo
  [authors]: R. Remington, The University of Florida
  [description]: Algorithm to calculate quantities relevant to HcalHaloData object
  [date]: October 15, 2009
*/

#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/METReco/interface/HaloClusterCandidateHCAL.h"



class HcalHaloAlgo{
 public:
  //constructor
  HcalHaloAlgo();
  //destructor
  ~HcalHaloAlgo(){}
  
  // run algorithm
  reco::HcalHaloData Calculate(const CaloGeometry& TheCaloGeometry, edm::Handle<HBHERecHitCollection>& TheHBHERecHits, edm::Handle<CaloTowerCollection>& TheCaloTowers,edm::Handle<EBRecHitCollection>& TheEBRecHits,edm::Handle<EERecHitCollection>& TheEERecHits,const edm::EventSetup& TheSetup);

  reco::HcalHaloData Calculate(const CaloGeometry& TheCaloGeometry, edm::Handle<HBHERecHitCollection>& TheHBHERecHits,edm::Handle<EBRecHitCollection>& TheEBRecHits,edm::Handle<EERecHitCollection>& TheEERecHits,const edm::EventSetup& TheSetup);
  
  // Set RecHit Energy Thresholds
  void SetRecHitEnergyThresholds( float HB, float HE){ HBRecHitEnergyThreshold = HB; HERecHitEnergyThreshold = HE;}
  
  // Set Phi Wedge Thresholds
  void SetPhiWedgeEnergyThreshold( float SumE ){ SumEnergyThreshold = SumE ;}
  void SetPhiWedgeNHitsThreshold( int nhits ) { NHitsThreshold = nhits ; }
  void SetPhiWedgeThresholds(float SumE, int nhits) { SumEnergyThreshold = SumE ; NHitsThreshold = nhits ;}
  
  // Get RecHit Energy Threshold
  float GetHBRecHitEnergyThreshold(){ return HBRecHitEnergyThreshold;}
  float GetHERecHitEnergyThreshold(){ return HERecHitEnergyThreshold;}
  
  // Get Phi Wedge Threhsolds
  float GetPhiWedgeEnergyThreshold() { return SumEnergyThreshold;}
  int GetPhiWedgeNHitsThreshold() { return NHitsThreshold;}
  

  std::vector<reco::HaloClusterCandidateHCAL> GetHaloClusterCandidateHB(edm::Handle<EcalRecHitCollection>& ebrechitcoll, edm::Handle<HBHERecHitCollection>& hbherechitcoll,float et_thresh_seedrh);
  std::vector<reco::HaloClusterCandidateHCAL> GetHaloClusterCandidateHE(edm::Handle<EcalRecHitCollection>& eerechitcoll, edm::Handle<HBHERecHitCollection>& hbherechitcoll,float et_thresh_seedrh);
  bool HBClusterShapeandTimeStudy(reco::HaloClusterCandidateHCAL hcand, bool ishlt);
  bool HEClusterShapeandTimeStudy(reco::HaloClusterCandidateHCAL hcand, bool ishlt);



 private:
  // Invidiual RecHit Threhsolds
  float HBRecHitEnergyThreshold;
  float HERecHitEnergyThreshold;
  
  // Phi Wedge Thresholds
  float SumEnergyThreshold;
  int NHitsThreshold;

  const CaloGeometry *geo;
  math::XYZPoint getPosition(const DetId &id, reco::Vertex::Point vtx);
  
};

#endif
