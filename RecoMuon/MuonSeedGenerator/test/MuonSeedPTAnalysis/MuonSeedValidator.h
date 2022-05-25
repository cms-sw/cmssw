#ifndef RecoMuon_MuonSeedValidator_H
#define RecoMuon_MuonSeedValidator_H

/** \class SeedValidator
 *
 *  Author: S.C. Kao  - UC Riverside
 */

//#include "RecoMuon/SeedGenerator/test/MuonSeedPTAnalysis/SegSelector.h"
#include "MuonSeedValidatorHisto.h"
#include "SegSelector.h"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <DataFormats/Common/interface/Handle.h>
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4D.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment2D.h>
#include <DataFormats/DTRecHit/interface/DTRecHitCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecHit1D.h>
#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/TrackReco/interface/TrackFwd.h>
#include <DataFormats/TrackReco/interface/TrackExtra.h>
#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>
#include <DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h>
#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>
#include <DataFormats/TrackingRecHit/interface/TrackingRecHit.h>

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/DTGeometry/interface/DTChamber.h>
#include <Geometry/DTGeometry/interface/DTLayer.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
//#include <Geometry/Records/interface/GlobalTrackingGeometryRecord.h>
#include <RecoMuon/TrackingTools/interface/MuonServiceProxy.h>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include <vector>
#include <map>
#include <string>
#include <utility>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

//class PSimHit;
class TFile;
class CSCLayer;
class CSCDetId;
class DTLayerId;
class DTSuperLayerId;
class DTChamberId;
class SegSelector;
//class MuonSeedBuilder;

class MuonSeedValidator : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  MuonSeedValidator(const edm::ParameterSet &pset);

  /// Destructor
  virtual ~MuonSeedValidator();

  // Operations
  /// Perform the real analysis
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup);

protected:
private:
  SegSelector *recsegSelector;
  MuonServiceProxy *theService;

  // Utility functions

  void CSCsegment_stat(edm::Handle<CSCSegmentCollection> cscSeg,
                       edm::ESHandle<CSCGeometry> cscGeom,
                       double trkEta,
                       double trkPhi);
  void DTsegment_stat(edm::Handle<DTRecSegment4DCollection> dtSeg,
                      edm::ESHandle<DTGeometry> dtGeom,
                      double trkEta,
                      double trkPhi);
  void Simsegment_stat(std::vector<SimSegment> sCSC_v, std::vector<SimSegment> sDT_v);

  void CSCRecHit_Stat(edm::Handle<CSCRecHit2DCollection> cscrechit,
                      edm::ESHandle<CSCGeometry> cscGeom,
                      double trkEta,
                      double trkPhi);
  void DTRecHit_Stat(edm::Handle<DTRecHitCollection> dtrechit,
                     edm::ESHandle<DTGeometry> dtGeom,
                     double trkEta,
                     double trkPhi);

  int ChargeAssignment(GlobalVector Va, GlobalVector Vb);

  void RecSeedReader(edm::Handle<TrajectorySeedCollection> rec_seeds);
  void SegOfRecSeed(edm::Handle<TrajectorySeedCollection> rec_seeds,
                    int seed_idx,
                    std::vector<SimSegment> sCSC_v,
                    std::vector<SimSegment> sDT_v);
  void SegOfRecSeed(edm::Handle<TrajectorySeedCollection> rec_seeds, int seed_idx);

  void StaTrackReader(edm::Handle<reco::TrackCollection> sta_trk, int sta_glb);
  void SimInfo(const edm::Handle<edm::SimTrackContainer> simTracks,
               const edm::Handle<edm::PSimHitContainer> dsimHits,
               const edm::Handle<edm::PSimHitContainer> csimHits,
               edm::ESHandle<DTGeometry> dtGeom,
               edm::ESHandle<CSCGeometry> cscGeom);
  int RecSegReader(edm::Handle<CSCSegmentCollection> cscSeg,
                   edm::Handle<DTRecSegment4DCollection> dtSeg,
                   edm::ESHandle<CSCGeometry> cscGeom,
                   edm::ESHandle<DTGeometry> dtGeom,
                   double trkEta,
                   double trkPhi);

  double getEta(double vx, double vy, double vz);
  double getEta(double theta);

  std::vector<int> IdentifyShowering(edm::Handle<CSCSegmentCollection> cscSeg,
                                     edm::ESHandle<CSCGeometry> cscGeom,
                                     edm::Handle<DTRecSegment4DCollection> dtSeg,
                                     edm::ESHandle<DTGeometry> dtGeom,
                                     double trkTheta,
                                     double trkPhi);
  double getdR(std::vector<double> etav, std::vector<double> phiv);

  // Histograms
  H2DRecHit1 *h_all;
  H2DRecHit2 *h_NoSeed;
  H2DRecHit3 *h_NoSta;
  H2DRecHit4 *h_Scope;
  H2DRecHit5 *h_UnRel;

  // The file which will store the histos
  TFile *theFile;

  //cscsegment_stat output
  int cscseg_stat[6];
  int cscseg_stat1[6];

  //dtsegment_stat output
  int dtseg_stat[6];
  int dtseg_stat1[6];

  //sim segment_stat output
  int simcscseg[6];
  int simdtseg[6];
  int simseg_sum;
  double simseg_eta;

  // RecHit_stat
  int cscrh_sum[6];
  int dtrh_sum[6];

  // reco-seeding reader
  int nu_seed;
  std::vector<int> nSegInSeed;
  std::vector<GlobalPoint> seed_gp;
  std::vector<GlobalVector> seed_gm;
  std::vector<LocalVector> seed_lv;
  std::vector<LocalPoint> seed_lp;

  std::vector<double> qbp;
  std::vector<double> qbpt;
  std::vector<double> err_qbp;
  std::vector<double> err_qbpt;
  std::vector<double> seed_mT;
  std::vector<double> seed_mA;
  std::vector<int> seed_layer;
  std::vector<double> err_dx;
  std::vector<double> err_dy;
  std::vector<double> err_x;
  std::vector<double> err_y;

  // seg info from seed
  std::vector<double> d_h;
  std::vector<double> d_f;
  std::vector<double> d_x;
  std::vector<double> d_y;
  std::vector<DetId> geoID;

  //  track reader
  int nu_sta;
  // inner position of sta
  std::vector<double> sta_phiP;
  std::vector<double> sta_thetaP;
  // sta vector information
  std::vector<GlobalVector> sta_gm;
  std::vector<double> sta_qbp;
  std::vector<double> sta_qbpt;
  std::vector<double> sta_thetaV;
  std::vector<double> sta_phiV;
  std::vector<double> sta_mT;
  std::vector<double> sta_mA;
  std::vector<double> sta_chi2;
  std::vector<int> sta_nHits;

  // Reco Segment Reader
  double ave_phi;
  double ave_eta;
  std::vector<double> phi_resid;
  std::vector<double> eta_resid;
  std::vector<double> dx_error;
  std::vector<double> dy_error;
  std::vector<double> x_error;
  std::vector<double> y_error;

  // Sim info
  typedef std::vector<double> layer_pt;
  typedef std::vector<double> layer_pa;

  std::vector<double> theta_v;
  std::vector<double> theta_p;
  std::vector<double> phi_v;
  std::vector<double> phi_p;

  std::vector<double> eta_trk;
  std::vector<double> theta_trk;
  std::vector<double> phi_trk;

  std::vector<double> theQ;
  std::vector<double> pt_trk;
  std::vector<layer_pt> ptlayer;
  std::vector<layer_pa> palayer;
  std::vector<int> trackID;

  // showering info
  std::vector<double> muCone;

  // Switch for debug output
  std::string rootFileName;
  std::string cscSegmentLabel;
  std::string recHitLabel;
  std::string dtSegmentLabel;
  std::string dtrecHitLabel;
  std::string simHitLabel;
  std::string simTrackLabel;
  std::string muonseedLabel;
  edm::InputTag staTrackLabel;
  edm::InputTag glbTrackLabel;

  bool debug;
  double dtMax;
  double dfMax;

  bool scope;
  double pTCutMax;
  double pTCutMin;
  double eta_Low;
  double eta_High;

  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken;
};

#endif
