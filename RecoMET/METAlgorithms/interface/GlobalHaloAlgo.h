#ifndef RECOMET_METALGORITHMS_GLOBALHALOALGO_H
#define RECOMET_METALGORITHMS_GLOBALHALOALGO_H

/*
  [class]:  GlobalHaloAlgo
  [authors]: R. Remington, The University of Florida
  [description]: Algorithm to calculate quantities relevant to GlobalHaloData object
  [date]: October 15, 2009
*/

#include "DataFormats/METReco/interface/EcalHaloData.h"
#include "DataFormats/METReco/interface/HcalHaloData.h"
#include "DataFormats/METReco/interface/CSCHaloData.h"
#include "DataFormats/METReco/interface/GlobalHaloData.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

class GlobalHaloAlgo {
 public: 
  // Constructor
  GlobalHaloAlgo();
  // Destructor
  ~GlobalHaloAlgo(){}
  
  // run algorithm
  reco::GlobalHaloData Calculate(const CaloGeometry& TheCaloGeometry, const CSCGeometry& TheCSCGeometry,const reco::CaloMET& TheCaloMET,edm::Handle<edm::View<reco::Candidate> >& TheCaloTowers, edm::Handle<CSCSegmentCollection>& TheCSCSegments, edm::Handle<CSCRecHit2DCollection>& TheCSCRecHits, const reco::CSCHaloData& TheCSCHaloData ,const reco::EcalHaloData& TheEcalHaloData, const reco::HcalHaloData& TheHcalHaloData); 
  
  // Set min & max radius to associate CSC Rechits with Ecal Phi Wedges
  void SetEcalMatchingRadius(float min, float max){Ecal_R_Min = min ; Ecal_R_Max = max;}
  // Set min & max radius to associate CSC Rechits with Hcal Phi Wedges
  void SetHcalMatchingRadius(float min, float max){Hcal_R_Min = min ; Hcal_R_Max = max;}
  // Set CaloTowerEtTheshold
  void SetCaloTowerEtThreshold(float EtMin) { TowerEtThreshold = EtMin ;}
  // run algorithm
  
 private:
  float Ecal_R_Min;
  float Ecal_R_Max;
  float Hcal_R_Min;
  float Hcal_R_Max;
  float TowerEtThreshold;
};

#endif
