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

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class GlobalHaloAlgo {
 public: 
  // Constructor
  GlobalHaloAlgo();
  // Destructor
  ~GlobalHaloAlgo(){}
  
  // run algorithm
  reco::GlobalHaloData Calculate(const CaloGeometry& TheCaloGeometry, const CSCGeometry& TheCSCGeometry,const reco::CaloMET& TheCaloMET,edm::Handle<edm::View<reco::Candidate> >& TheCaloTowers, edm::Handle<CSCSegmentCollection>& TheCSCSegments, edm::Handle<CSCRecHit2DCollection>& TheCSCRecHits, edm::Handle<reco::MuonCollection>& TheMuons, const reco::CSCHaloData& TheCSCHaloData ,const reco::EcalHaloData& TheEcalHaloData, const reco::HcalHaloData& TheHcalHaloData, bool ishlt =false); 
  
  // Set min & max radius to associate CSC Rechits with Ecal Phi Wedges
  void SetEcalMatchingRadius(float min, float max){Ecal_R_Min = min ; Ecal_R_Max = max;}
  // Set min & max radius to associate CSC Rechits with Hcal Phi Wedges
  void SetHcalMatchingRadius(float min, float max){Hcal_R_Min = min ; Hcal_R_Max = max;}
  // Set CaloTowerEtTheshold
  void SetCaloTowerEtThreshold(float EtMin) { TowerEtThreshold = EtMin ;}
  // run algorithm

  //CSC-Calo matching parameters:
  void SetMaxSegmentTheta(float x) { max_segment_theta = x; }
  //EB  
  void setEtThresholdforCSCCaloMatchingEB(float x ){et_thresh_rh_eb= x;}
  void setRcaloMinRsegmLowThresholdforCSCCaloMatchingEB(float x ){dr_lowthresh_segvsrh_eb= x;}
  void setRcaloMinRsegmHighThresholdforCSCCaloMatchingEB(float x ){dr_highthresh_segvsrh_eb= x;}
  void setDtcalosegmThresholdforCSCCaloMatchingEB(float x ){dt_segvsrh_eb= x;}
  void setDPhicalosegmThresholdforCSCCaloMatchingEB(float x ){dphi_thresh_segvsrh_eb= x;}
  //EE  
  void setEtThresholdforCSCCaloMatchingEE(float x ){et_thresh_rh_ee= x;}
  void setRcaloMinRsegmLowThresholdforCSCCaloMatchingEE(float x ){dr_lowthresh_segvsrh_ee= x;}
  void setRcaloMinRsegmHighThresholdforCSCCaloMatchingEE(float x ){dr_highthresh_segvsrh_ee= x;}
  void setDtcalosegmThresholdforCSCCaloMatchingEE(float x ){dt_segvsrh_ee= x;}
  void setDPhicalosegmThresholdforCSCCaloMatchingEE(float x ){dphi_thresh_segvsrh_ee= x;}
  //HB  
  void setEtThresholdforCSCCaloMatchingHB(float x ){et_thresh_rh_hb= x;}
  void setRcaloMinRsegmLowThresholdforCSCCaloMatchingHB(float x ){dr_lowthresh_segvsrh_hb= x;}
  void setRcaloMinRsegmHighThresholdforCSCCaloMatchingHB(float x ){dr_highthresh_segvsrh_hb= x;}
  void setDtcalosegmThresholdforCSCCaloMatchingHB(float x ){dt_segvsrh_hb= x;}
  void setDPhicalosegmThresholdforCSCCaloMatchingHB(float x ){dphi_thresh_segvsrh_hb= x;}
  //HE  
  void setEtThresholdforCSCCaloMatchingHE(float x ){et_thresh_rh_he= x;}
  void setRcaloMinRsegmLowThresholdforCSCCaloMatchingHE(float x ){dr_lowthresh_segvsrh_he= x;}
  void setRcaloMinRsegmHighThresholdforCSCCaloMatchingHE(float x ){dr_highthresh_segvsrh_he= x;}
  void setDtcalosegmThresholdforCSCCaloMatchingHE(float x ){dt_segvsrh_he= x;}
  void setDPhicalosegmThresholdforCSCCaloMatchingHE(float x ){dphi_thresh_segvsrh_he= x;}


  
 private:
  float Ecal_R_Min;
  float Ecal_R_Max;
  float Hcal_R_Min;
  float Hcal_R_Max;
  float TowerEtThreshold;

  //Parameters for CSC-calo matching
  float max_segment_theta;

  float  et_thresh_rh_eb;
  float dphi_thresh_segvsrh_eb;
  float dr_lowthresh_segvsrh_eb;
  float dr_highthresh_segvsrh_eb;
  float dt_segvsrh_eb;

  float  et_thresh_rh_ee;
  float dphi_thresh_segvsrh_ee;
  float dr_lowthresh_segvsrh_ee;
  float dr_highthresh_segvsrh_ee;
  float dt_segvsrh_ee;

  float  et_thresh_rh_hb;
  float dphi_thresh_segvsrh_hb;
  float dr_lowthresh_segvsrh_hb;
  float dr_highthresh_segvsrh_hb;
  float dt_segvsrh_hb;

  float  et_thresh_rh_he;
  float dphi_thresh_segvsrh_he;
  float dr_lowthresh_segvsrh_he;
  float dr_highthresh_segvsrh_he;
  float dt_segvsrh_he;



  void AddtoBeamHaloEBEERechits(edm::RefVector<EcalRecHitCollection>& bhtaggedrechits,reco::GlobalHaloData & thehalodata, bool isbarrel);
  void AddtoBeamHaloHBHERechits(edm::RefVector<HBHERecHitCollection>& bhtaggedrechits,reco::GlobalHaloData & thehalodata);
  bool SegmentMatchingEB(reco::GlobalHaloData & thehalodata, const std::vector<reco::HaloClusterCandidateECAL> & haloclustercands, float iZ, float iR, float iT, float iPhi, bool ishlt);
  bool SegmentMatchingEE(reco::GlobalHaloData & thehalodata, const std::vector<reco::HaloClusterCandidateECAL> & haloclustercands, float iZ, float iR, float iT, float iPhi, bool ishlt);
  bool SegmentMatchingHB(reco::GlobalHaloData & thehalodata, const std::vector<reco::HaloClusterCandidateHCAL> & haloclustercands, float iZ, float iR, float iT, float iPhi, bool ishlt);
  bool SegmentMatchingHE(reco::GlobalHaloData & thehalodata, const std::vector<reco::HaloClusterCandidateHCAL> & haloclustercands, float iZ, float iR, float iT, float iPhi, bool ishlt);
  bool ApplyMatchingCuts(int subdet, bool ishlt, double rhet, double segZ, double rhZ, double segR, double rhR, double segT, double rhT, double segPhi, double rhPhi);

};

#endif
