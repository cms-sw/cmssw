#ifndef RECOMET_METALGORITHMS_CSCHALOALGO_H
#define RECOMET_METALGORITHMS_CSCHALOALGO_H
#include "DataFormats/METReco/interface/CSCHaloData.h"

/*
  [class]:  CSCHaloAlgo
  [authors]: R. Remington, The University of Florida
  [description]: Algorithm to calculate quantities relevant to CSCHaloData object
  [date]: October 15, 2009
*/

#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonSegmentMatcher.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/Event.h"


namespace edm {
  class TriggerNames;
}

class CSCHaloAlgo {

 public:
  CSCHaloAlgo();
  ~CSCHaloAlgo(){}
  reco::CSCHaloData Calculate(const CSCGeometry& TheCSCGeometry,edm::Handle<reco::MuonCollection>& TheCosmicMuons, 
			      const edm::Handle<reco::MuonTimeExtraMap> TheCSCTimeMap,
			      edm::Handle<reco::MuonCollection>& TheMuons, edm::Handle<CSCSegmentCollection>& TheCSCSegments, 
			      edm::Handle<CSCRecHit2DCollection>& TheCSCRecHits,edm::Handle < L1MuGMTReadoutCollection >& TheL1GMTReadout,
			      edm::Handle<HBHERecHitCollection>& hbhehits,edm::Handle<EcalRecHitCollection>& ecalebhits,
			      edm::Handle<EcalRecHitCollection>& ecaleehits,
			      edm::Handle<edm::TriggerResults>& TheHLTResults, const edm::TriggerNames * triggerNames, 
			      const edm::Handle<CSCALCTDigiCollection>& TheALCTs, MuonSegmentMatcher *TheMatcher,
			      const edm::Event &TheEvent, const edm::EventSetup &TheEventSetup);

  std::vector<edm::InputTag> vIT_HLTBit;

  void SetDetaThreshold(float x ){ deta_threshold = x;}
  void SetMinMaxInnerRadius(float min, float max){ min_inner_radius = min; max_inner_radius = max;}
  void SetMinMaxOuterRadius(float min, float max) { min_outer_radius = min; max_outer_radius = max;}
  void SetDphiThreshold(float x) { dphi_threshold = x;}
  void SetNormChi2Threshold(float x) { norm_chi2_threshold = x;}
  void SetRecHitTime0(float x) { recHit_t0 = x;}
  void SetRecHitTimeWindow(float x) { recHit_twindow = x; }
  void SetExpectedBX(int x) { expected_BX = x ;}
  void SetMinMaxOuterMomentumTheta(float min , float max){ min_outer_theta = min;  max_outer_theta = max;}
  void SetMatchingDPhiThreshold(float x) { matching_dphi_threshold = x;}
  void SetMatchingDEtaThreshold(float x) { matching_deta_threshold = x;}
  void SetMatchingDWireThreshold(int x) { matching_dwire_threshold = x;}
  void SetMaxDtMuonSegment(float x ){ max_dt_muon_segment = x;}
  void SetMaxFreeInverseBeta(float x ){ max_free_inverse_beta = x;}

  // MLR
  void SetMaxSegmentRDiff(float x) { max_segment_r_diff = x; }
  void SetMaxSegmentPhiDiff(float x) { max_segment_phi_diff = x; }
  void SetMaxSegmentTheta(float x) { max_segment_theta = x; }
  // End MLR

 private:
  float deta_threshold;
  float max_outer_theta;
  float min_outer_theta;
  float min_inner_radius;
  float max_inner_radius;
  float min_outer_radius;
  float max_outer_radius;
  float dphi_threshold;
  float norm_chi2_threshold;
  float recHit_t0;
  float recHit_twindow;
  int expected_BX;
  float matching_dphi_threshold;
  float matching_deta_threshold;
  int matching_dwire_threshold;
  float max_dt_muon_segment;
  float max_free_inverse_beta;
  // MLR
  float max_segment_r_diff;
  float max_segment_phi_diff;
  float max_segment_theta;
  // End MLR
  float  et_thresh_rh_hbhe, dphi_thresh_segvsrh_hbhe,dr_lowthresh_segvsrh_hbhe, dr_highthresh_segvsrh_hbhe, dt_lowthresh_segvsrh_hbhe;
  float  et_thresh_rh_eb, dphi_thresh_segvsrh_eb,dr_lowthresh_segvsrh_eb, dr_highthresh_segvsrh_eb, dt_lowthresh_segvsrh_eb;
  float  et_thresh_rh_ee, dphi_thresh_segvsrh_ee,dr_lowthresh_segvsrh_ee, dr_highthresh_segvsrh_ee, dt_lowthresh_segvsrh_ee;

  
  
  const CaloGeometry *geo;
  math::XYZPoint getPosition(const DetId &id, reco::Vertex::Point vtx);
  bool HCALSegmentMatching(edm::Handle<HBHERecHitCollection>& rechitcoll, float et_thresh_rh, float dphi_thresh_segvsrh, float dr_lowthresh_segvsrh, float dr_highthresh_segvsrh, float dt_lowthresh_segvsrh , float iZ, float iR, float iT, float iPhi);
  bool ECALSegmentMatching(edm::Handle<EcalRecHitCollection>& rechitcoll,  float et_thresh_rh, float dphi_thresh_segvsrh, float dr_lowthresh_segvsrh, float dr_highthresh_segvsrh, float dt_lowthresh_segvsrh, float iZ, float iR, float iT, float iPhi );

};

#endif
