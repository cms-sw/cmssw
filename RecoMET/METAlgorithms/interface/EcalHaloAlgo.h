#ifndef RECOMET_METALGORITHMS_ECALHALOALGO_H
#define RECOMET_METALGORITHMS_ECALHALOALGO_H
/*
  [class]:  EcalHaloAlgo
  [authors]: R. Remington, The University of Florida
  [description]: Algorithm to calculate quantities relevant to EcalHaloData object
  [date]: October 15, 2009
*/
#include "DataFormats/METReco/interface/EcalHaloData.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
//#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitDefs.h"
#include "DataFormats/METReco/interface/HaloClusterCandidateECAL.h"

class EcalHaloAlgo {
public:
  // constructor
  explicit EcalHaloAlgo(edm::ConsumesCollector iC);
  // destructor
  ~EcalHaloAlgo() {}

  // Run algorithm
  reco::EcalHaloData Calculate(const CaloGeometry& TheCaloGeometry,
                               edm::Handle<reco::PhotonCollection>& ThePhotons,
                               edm::Handle<reco::SuperClusterCollection>& TheSuperClusters,
                               edm::Handle<EBRecHitCollection>& TheEBRecHits,
                               edm::Handle<EERecHitCollection>& TheEERecHits,
                               edm::Handle<ESRecHitCollection>& TheESRecHits,
                               edm::Handle<HBHERecHitCollection>& TheHBHERecHits,
                               const edm::EventSetup& TheSetup);
  // Set Roundness cuts
  void SetRoundnessCut(float r = 100.) { RoundnessCut = r; }
  // Set Angle cuts
  void SetAngleCut(float a = 4.) { AngleCut = a; }

  // Set RecHit Energy Thresholds
  void SetRecHitEnergyThresholds(float EB, float EE, float ES) {
    EBRecHitEnergyThreshold = EB;
    EERecHitEnergyThreshold = EE;
    ESRecHitEnergyThreshold = ES;
  }
  // Set Phi Wedge Thresholds
  void SetPhiWedgeEnergyThreshold(float SumE) { SumEnergyThreshold = SumE; }
  void SetPhiWedgeNHitsThreshold(int nhits) { NHitsThreshold = nhits; }
  void SetPhiWedgeThresholds(float SumE, int nhits) {
    SumEnergyThreshold = SumE;
    NHitsThreshold = nhits;
  }

  std::vector<reco::HaloClusterCandidateECAL> GetHaloClusterCandidateEB(
      edm::Handle<EcalRecHitCollection>& ecalrechitcoll,
      edm::Handle<HBHERecHitCollection>& hbherechitcoll,
      float et_thresh_seedrh);
  std::vector<reco::HaloClusterCandidateECAL> GetHaloClusterCandidateEE(
      edm::Handle<EcalRecHitCollection>& ecalrechitcoll,
      edm::Handle<HBHERecHitCollection>& hbherechitcoll,
      float et_thresh_seedrh);
  bool EBClusterShapeandTimeStudy(reco::HaloClusterCandidateECAL hcand, bool ishlt);
  bool EEClusterShapeandTimeStudy_ITBH(reco::HaloClusterCandidateECAL hcand, bool ishlt);
  bool EEClusterShapeandTimeStudy_OTBH(reco::HaloClusterCandidateECAL hcand, bool ishlt);

  // Get Roundness cut
  float GetRoundnessCut() { return RoundnessCut; }
  // Get Angle cut
  float GetAngleCut() { return AngleCut; }

  // Get RecHit Energy Threshold
  float GetEBRecHitEnergyThreshold() { return EBRecHitEnergyThreshold; }
  float GetEERecHitEnergyThreshold() { return EERecHitEnergyThreshold; }
  float GetESRecHitEnergyThreshold() { return ESRecHitEnergyThreshold; }

  // Get Phi Wedge Threhsolds
  float GetPhiWedgeEnergyThreshold() { return SumEnergyThreshold; }
  int GetPhiWedgeNHitsThreshold() { return NHitsThreshold; }

private:
  // Cut Value for Supercluster "roundness" variable
  float RoundnessCut;
  // Cut Value for Supercluster "angle" variable
  float AngleCut;

  // Invidiual RecHit Threhsolds
  float EBRecHitEnergyThreshold;
  float EERecHitEnergyThreshold;
  float ESRecHitEnergyThreshold;

  // Phi Wedge Thresholds
  float SumEnergyThreshold;
  int NHitsThreshold;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geoToken_;
  const CaloGeometry* geo;
  math::XYZPoint getPosition(const DetId& id, reco::Vertex::Point vtx);
};

#endif
