#ifndef RECOMET_METALGORITHMS_HCALHALOALGO_H
#define RECOMET_METALGORITHMS_HCALHALOALGO_H

#include "DataFormats/METReco/interface/HcalHaloData.h"

/*
  [class]:  HcalHaloAlgo
  [authors]: R. Remington, The University of Florida
  [description]: Algorithm to calculate quantities relevant to HcalHaloData object
  [date]: October 15, 2009
*/

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

class HcalHaloAlgo{
 public:
  //constructor
  HcalHaloAlgo();
  //destructor
  ~HcalHaloAlgo(){}
  
  // run algorithm
  reco::HcalHaloData Calculate(const CaloGeometry& TheCaloGeometry, edm::Handle<HBHERecHitCollection>& TheHBHERecHits, edm::Handle<CaloTowerCollection>& TheCaloTowers);

  reco::HcalHaloData Calculate(const CaloGeometry& TheCaloGeometry, edm::Handle<HBHERecHitCollection>& TheHBHERecHits);
  
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
  
 private:
  // Invidiual RecHit Threhsolds
  float HBRecHitEnergyThreshold;
  float HERecHitEnergyThreshold;
  
  // Phi Wedge Thresholds
  float SumEnergyThreshold;
  int NHitsThreshold;
  
};

#endif
