#ifndef CSC_HALO_FILTER_H
#define CSC_HALO_FILTER_H

#include "FWCore/Framework/interface/EDFilter.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/TriggerResults.h"
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
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "DataFormats/METReco/interface/BeamHaloSummary.h"
//Root Classes

#include "TH1F.h"
#include "TH2F.h"
#include "TH1I.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TString.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "TLegend.h"

//Standard C++ classes
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>

class CSCHaloFilter : public edm::EDFilter {
 public:
  
  explicit CSCHaloFilter(const edm::ParameterSet & iConfig);
  virtual ~CSCHaloFilter();
  
 private:
  
  virtual bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup);

  edm::InputTag IT_L1MuGMTReadout;
  edm::InputTag IT_ALCTDigi;
  edm::InputTag IT_CollisionMuon;
  edm::InputTag IT_SACosmicMuon;
  edm::InputTag IT_CSCRecHit;
  edm::InputTag IT_CSCSegment;
  edm::InputTag IT_CSCHaloData;
  edm::InputTag IT_BeamHaloSummary;
  bool FilterCSCLoose;
  bool FilterCSCTight;

  bool FilterDigiLevel;    //requires CSCALCTDigiCollection  (usually not available in RECO data tier)
  bool FilterTriggerLevel; //requires L1MuGMTReadoutCollection
  bool FilterRecoLevel;    //requires Cosmic reco::TrackCollection

  //min value of deta between innermost and outermost hit of cosmic reco::Track in CSCs
  float deta_threshold;  
  //max value of dphi between innermost and outermost hit of cosmic reco::Track in CSCs
  float dphi_threshold;
  //max value of outer-momentum theta for cosmic reco::Track in CSCs
  float max_outer_theta;
  //min value of outer-momentum theta for cosmic reco::Track in CSCs
  float min_outer_theta;
  //min value of innermost constituent rechit radius for cosmic reco::Track in CSCs
  float min_inner_radius;
  //max value of innermost constituent rechit radius for cosmic reco::Track in CSCs
  float max_inner_radius;
  //min value of outermost constituent rechit radius for cosmic reco::Track in CSCs
  float min_outer_radius;
  //max value of outermost constituent rechit radius for cosmic reco::Track in CSCs
  float max_outer_radius;
  //threshold on chi2 of cosmic reco::Track in CSCs 
  float norm_chi2_threshold;
  //max value of dr/dz calculated using innermost and outermose rechit from cosmic reco::Track in CSCs
  float max_dr_over_dz;

  //expected local BX number of ALCT Digi for collision induced LCTs (3 for Data, 6 for MC)
  int expected_BX;

  //dphi window for matching collision muon hits to L1 halo candidates
  float matching_dphi_threshold;
  //deta window for matching collision muon hits to L1 halo candidates
  float matching_deta_threshold;
  //dwire window for matching collision muon hits to early ALCT Digis
  int matching_dwire_threshold; 

  int min_nHaloDigis;
  int min_nHaloTriggers;
  int min_nHaloTracks; 
  
  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters parameters_;



};

#endif
