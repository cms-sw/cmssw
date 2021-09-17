// -*- C++ -*-
//
// Package:  Triplet
// Class:    Triplet
//
// my/Triplet/src/Triplet.cc
//
// plot hits on CMS tracks on RECO
//
// Original Author:  Daniel Pitzl, DESY,,
//         Created:  Sat Feb 12 12:12:42 CET 2011
// $Id$
// d.k.
// Split into a sperate call.
//

// system include files:
#include <memory>
#include <iostream>
#include <cmath>

// ROOT:
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"

// CMS and user include files:
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
//#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/MakerMacros.h"

#include <DataFormats/BeamSpot/interface/BeamSpot.h>

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

//#include "DataFormats/METReco/interface/PFMET.h"
//#include "DataFormats/METReco/interface/PFMETFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <DataFormats/TrackReco/interface/HitPattern.h>

// To convert detId to subdet/layer number:
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
//#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include <TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h>
#include <MagneticField/Engine/interface/MagneticField.h>

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include <TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h>
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

//#define SIM // use for single muon simulations

//
// class declaration:
//
class myCounters {
public:
  static int neve;
  static unsigned int prevrun;
};

int myCounters::neve = 0;
unsigned int myCounters::prevrun = 0;
//
//
//
class Triplet : public edm::EDAnalyzer {
public:
  explicit Triplet(const edm::ParameterSet &);
  ~Triplet();

private:
  virtual void beginJob();
  virtual void analyze(const edm::Event &, const edm::EventSetup &);
  virtual void endJob();
  void triplets(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2,
                double x3,
                double y3,
                double z3,
                double ptsig,
                double &dc,
                double &dz);

  // ----------member data:
  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::EDGetTokenT<reco::TrackCollection> trackToken_;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackBuilderToken_;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> transientTrackingRecHitBuilderToken_;

  TH1D *h000, *h001, *h002, *h003, *h004, *h005, *h006, *h007, *h008, *h009;
  TH1D *h010, *h011, *h012, *h013, *h014, *h015, *h016, *h017, *h018, *h019;
  TH1D *h020, *h021, *h022, *h023, *h024, *h025, *h026, *h027, *h028, *h029;
  TH1D *h030, *h031, *h032, *h033, *h034, *h035, *h036, *h037, *h038, *h039;
  TH1D *h040, *h041, *h042, *h043, *h044, *h045, *h046, *h047, *h048, *h049;
  TH1D *h050, *h051, *h052, *h053, *h054, *h055, *h056, *h057, *h058, *h059;
  TH1D *h060, *h061, *h062, *h063, *h064, *h065, *h067, *h068, *h069;
  TH2D *h066;
  TH1D *h070, *h071, *h072, *h073, *h074, *h075, *h076, *h077, *h078, *h079;
  TH1D *h080, *h081, *h082, *h083, *h084, *h085, *h086, *h087, *h088, *h089;
  TH1D *h090, *h091, *h092, *h093, *h094;
  TH2D *h095, *h096, *h097, *h098, *h099;

  TH1D *h100, *h101, *h102, *h103, *h104, *h105, *h108;
  TH2D *h106, *h107, *h109;
  TH1D *h110, *h112, *h113, *h114, *h115, *h116, *h118, *h119;
  TH2D *h111, *h117;
  TH1D *h120, *h121, *h122, *h123, *h124, *h125, *h126, *h127, *h128, *h129;
  TH1D *h130, *h131, *h132, *h133, *h134, *h135, *h136, *h137, *h138, *h139;
  TH1D *h140, *h141, *h142, *h143, *h144, *h145, *h146, *h147, *h148, *h149;
  TH1D *h150, *h151, *h152, *h153, *h154, *h155, *h156, *h157, *h158, *h159;
  TH1D *h160, *h161, *h162, *h163, *h164, *h165, *h166, *h167, *h168, *h169;
  TH1D *h170, *h171, *h172, *h173, *h174, *h175, *h176, *h177, *h178, *h179;
  TH1D *h180, *h181, *h182, *h183, *h184, *h185, *h186, *h187, *h188, *h189;
  TH1D *h190, *h191, *h192, *h193, *h194, *h195, *h196, *h197, *h198, *h199;

  TH1D *h200, *h201, *h202, *h203, *h204, *h205, *h208;
  TH2D *h206, *h207, *h209;

  TH1D *h300, *h301, *h302, *h303, *h304, *h305, *h308;
  TH2D *h306, *h307, *h309;

  TH1D *h400, *h401, *h402, *h403, *h404, *h405, *h406, *h407, *h408;
  TProfile *h409;
  TH1D *h410, *h411;
  TProfile *h412, *h413, *h414, *h415, *h416, *h417, *h418, *h419;
  TH1D *h420, *h421;
  TProfile *h422, *h423, *h424, *h425, *h426, *h427, *h428, *h429;
  TH1D *h430, *h431, *h432, *h435, *h436, *h437, *h438, *h439;
  TProfile *h433, *h434;
  TH1D *h440, *h441;
  TProfile *h442, *h443, *h444, *h445, *h446, *h447, *h448, *h449;
  TH1D *h450, *h451;
};

//
// constants, enums and typedefs:
//

//
// static data member definitions:
//

//
// constructor:
//
Triplet::Triplet(const edm::ParameterSet &iConfig) {
  bsToken_ = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  vtxToken_ = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  trackToken_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));

  trackerTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  transientTrackBuilderToken_ =
      esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"));
  transientTrackingRecHitBuilderToken_ =
      esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(edm::ESInputTag("", "WithTrackAngle"));

  std::cout << "Triplet constructed\n";
}
//
// destructor:
//
Triplet::~Triplet() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}
//
// member functions:
// method called once each job just before starting event loop
//
void Triplet::beginJob() {
  edm::Service<TFileService> fs;

  h000 = fs->make<TH1D>("h000", "beta star; beta star [cm]", 100, 0, 500);
  h001 = fs->make<TH1D>("h001", "emittance; emittance x [cm]", 100, 0, 1e-5);
  h002 = fs->make<TH1D>("h002", "beam width x; beam width x [um]", 100, 0, 200);
  h003 = fs->make<TH1D>("h003", "beam width y; beam width y [um]", 100, 0, 200);
  h004 = fs->make<TH1D>("h004", "beam spot sigma z; beam spot sigma z [cm]", 100, 0, 20);
  h005 = fs->make<TH1D>("h005", "beam spot x; beam spot x [cm]", 100, -1, 1);
  h006 = fs->make<TH1D>("h006", "beam spot y; beam spot y [cm]", 100, -1, 1);
  h007 = fs->make<TH1D>("h007", "beam spot z; beam spot z [cm]", 100, -5, 5);
  h008 = fs->make<TH1D>("h008", "beam slope dxdz; beam slope dxdz [mrad]", 100, -5, 5);
  h009 = fs->make<TH1D>("h009", "beam slope dydz; beam slope dydz [mrad]", 100, -5, 5);

  h010 = fs->make<TH1D>("h010", "number of primary vertices; vertices; events", 31, -0.5, 30.5);
  h011 = fs->make<TH1D>("h011", "invalid z-vertex;z [cm]", 100, -50, 50);
  h012 = fs->make<TH1D>("h012", "fake z-vertex;z [cm]", 100, -50, 50);
  h013 = fs->make<TH1D>("h013", "non-fake z-vertex;z [cm]", 100, -50, 50);
  h014 = fs->make<TH1D>("h014", "vertex x;x [cm]", 100, -0.5, 0.5);
  h015 = fs->make<TH1D>("h015", "vertex y;y [cm]", 100, -0.5, 0.5);
  h016 = fs->make<TH1D>("h016", "tracks per vertex; tracks; vertices", 101, -0.5, 100.5);
  h017 = fs->make<TH1D>("h017", "tracks per vertex; tracks; vertices", 100, 5, 505);
  h018 = fs->make<TH1D>("h018", "z-vertex with refitted tracks;z [cm]", 100, -50, 50);
  h019 = fs->make<TH1D>("h019", "z-vertex without refitted tracks;z [cm]", 100, -50, 50);

  h021 = fs->make<TH1D>("h021", "vertex sum pt; sum pt [GeV]", 100, 0, 100);
  h022 = fs->make<TH1D>("h022", "vertex max sum pt; max sum pt [GeV]", 100, 0, 100);

  h023 = fs->make<TH1D>("h023", "best vertex x;x [cm]", 100, -0.25, 0.25);
  h024 = fs->make<TH1D>("h024", "best vertex y;y [cm]", 100, -0.25, 0.25);
  h025 = fs->make<TH1D>("h025", "best vertex z;z [cm]", 100, -25, 25);

  //h026 = fs->make<TH1D>("h026", "Sum Et; Sum Et [GeV]", 100, 0, 1000 );
  //h027 = fs->make<TH1D>("h027", "MET; MET [GeV]", 100, 0, 200 );

  h028 = fs->make<TH1D>("h028", "sum track pt; sum track Pt [GeV]", 100, 0, 500);
  h029 = fs->make<TH1D>("h029", "sum primary track charge; sum track charge", 41, -20.5, 20.5);

  h040 = fs->make<TH1D>("h040", "number of tracks; tracks", 101, -5, 1005);
  h041 = fs->make<TH1D>("h041", "track charge; charge", 11, -5.5, 5.5);
  h042 = fs->make<TH1D>("h042", "pt; pt [GeV]", 100, 0, 5);
  h043 = fs->make<TH1D>("h043", "pt use logy, pt [GeV]", 100, 0, 100);

  h044 = fs->make<TH1D>("h044", "number of rec hits; hits; tracks", 51, -0.5, 50.5);
  h045 = fs->make<TH1D>("h045", "valid tracker hits; tracker hits; tracks", 51, -0.5, 50.5);
  h046 = fs->make<TH1D>("h046", "valid pixel barrel hits; valid pixel barrel hits; tracks", 11, -0.5, 10.5);
  h047 = fs->make<TH1D>("h047", "tracker layers; tracker layers; tracks", 31, -0.5, 30.5);
  h048 = fs->make<TH1D>("h048", "pixel barrel layers; pixel barrel layers; tracks", 11, -0.5, 10.5);

  h051 = fs->make<TH1D>("h051", "kap-kap; dkap; tracks", 100, -0.01, 0.01);
  h052 = fs->make<TH1D>("h052", "phi-phi; dphi; tracks", 100, -0.1, 0.1);
  h053 = fs->make<TH1D>("h053", "dca-dca; ddca; tracks", 100, -0.1, 0.1);
  h054 = fs->make<TH1D>("h054", "dip-dip; ddip; tracks", 100, -0.1, 0.1);
  h055 = fs->make<TH1D>("h055", "z0-z0; dz0; tracks", 100, -0.1, 0.1);

  h056 = fs->make<TH1D>("h056", "tracks", 100, 0.01142, 0.01143);
  h049 = fs->make<TH1D>("h049", "tracks", 1000, -0.000001, 0.000001);

  h057 = fs->make<TH1D>("h057", "tscp ref x; x [cm]; hits", 100, -1, 1);
  h058 = fs->make<TH1D>("h058", "tscp ref y; y [cm]; hits", 100, -1, 1);
  h059 = fs->make<TH1D>("h059", "tscp ref z; z [cm]; hits", 100, -10, 10);

  h060 = fs->make<TH1D>("h060", "rec hit tracker subdet; subdet ID; tracks", 11, -0.5, 10.5);
  h061 = fs->make<TH1D>("h061", "rec hits local x; x [cm]; hits", 120, -6, 6);
  h062 = fs->make<TH1D>("h062", "rec hits local y; y [cm]; hits", 80, -4, 4);

  h063 = fs->make<TH1D>("h063", "rec hits global R; R [cm]; hits", 120, 0, 120);
  h064 = fs->make<TH1D>("h064", "rec hits global phi; phi [deg]; hits", 180, -180, 180);
  h065 = fs->make<TH1D>("h065", "rec hits global z; z [cm]; hits", 300, -300, 300);

  h066 = fs->make<TH2D>("h066", "rec hits barrel x-y; x [cm]; y [cm]", 240, -120, 120, 240, -120, 120);

  h100 = fs->make<TH1D>("h100", "hits on tracks PXB layer; PXB layer; hits", 6, -0.5, 5.5);

  h101 = fs->make<TH1D>("h101", "hits on tracks PXB1 ladder; ladder; hits", 22, -0.5, 21.5);
  h102 = fs->make<TH1D>("h102", "hits on tracks PXB1 module; module; hits", 10, -0.5, 9.5);
  h103 = fs->make<TH1D>("h103", "hits on tracks PXB1 R; R [cm]; hits", 150, 0, 15);
  h104 = fs->make<TH1D>("h104", "hits on tracks PXB1 phi; phi [deg]; hits", 180, -180, 180);
  h105 = fs->make<TH1D>("h105", "hits on tracks PXB1 z; z [cm]; hits", 600, -30, 30);
  h106 = fs->make<TH2D>("h106", "hits on tracks PXB1 phi-z; phi [deg]; z [cm]", 180, -180, 180, 600, -30, 30);
  h107 = fs->make<TH2D>("h107", "hits local x-y PXB1; x [cm]; y [cm]", 180, -0.9, 0.9, 440, -3.3, 3.3);

  h111 = fs->make<TH2D>("h111", "hits on tracks PXB1 x-y; x [cm]; y [cm]", 240, -6, 6, 240, -6, 6);
  h112 = fs->make<TH1D>("h112", "residuals PXB1 dU; dU [um]; hits", 100, -250, 250);
  h113 = fs->make<TH1D>("h113", "residuals PXB1 dZ; dZ [um]; hits", 100, -250, 250);
  h114 = fs->make<TH1D>("h114", "residuals PXB1 dU; dU [um]; hits", 100, -250, 250);
  h115 = fs->make<TH1D>("h115", "residuals PXB1 dZ; dZ [um]; hits", 100, -250, 250);

  h201 = fs->make<TH1D>("h201", "hits on tracks PXB2 ladder; ladder; hits", 34, -0.5, 33.5);
  h202 = fs->make<TH1D>("h202", "hits on tracks PXB2 module; module; hits", 10, -0.5, 9.5);
  h203 = fs->make<TH1D>("h203", "hits on tracks PXB2 R; R [cm]; hits", 150, 0, 15);
  h204 = fs->make<TH1D>("h204", "hits on tracks PXB2 phi; phi [deg]; hits", 180, -180, 180);
  h205 = fs->make<TH1D>("h205", "hits on tracks PXB2 z; z [cm]; hits", 600, -30, 30);
  h206 = fs->make<TH2D>("h206", "hits on tracks PXB2 phi-z; phi [deg]; z [cm]", 180, -180, 180, 600, -30, 30);
  h207 = fs->make<TH2D>("h207", "hits local x-y PXB2; x [cm]; y [cm]", 180, -0.9, 0.9, 440, -3.3, 3.3);

  h301 = fs->make<TH1D>("h301", "hits on tracks PXB3 ladder; ladder; hits", 46, -0.5, 45.5);
  h302 = fs->make<TH1D>("h302", "hits on tracks PXB3 module; module; hits", 10, -0.5, 9.5);
  h303 = fs->make<TH1D>("h303", "hits on tracks PXB3 R; R [cm]; hits", 150, 0, 15);
  h304 = fs->make<TH1D>("h304", "hits on tracks PXB3 phi; phi [deg]; hits", 180, -180, 180);
  h305 = fs->make<TH1D>("h305", "hits on tracks PXB3 z; z [cm]; hits", 600, -30, 30);
  h306 = fs->make<TH2D>("h306", "hits on tracks PXB3 phi-z; phi [deg]; z [cm]", 180, -180, 180, 600, -30, 30);
  h307 = fs->make<TH2D>("h307", "hits local x-y PXB3; x [cm]; y [cm]", 180, -0.9, 0.9, 440, -3.3, 3.3);
  //
  // triplets:
  //
  h401 = fs->make<TH1D>("h401", "triplets z2; z [cm]; hits", 600, -30, 30);
  h402 = fs->make<TH1D>("h402", "uphi-phi; dphi; tracks", 100, -0.1, 0.1);
  h403 = fs->make<TH1D>("h403", "udca-dca; ddca; tracks", 100, -0.1, 0.1);
  h404 = fs->make<TH1D>("h404", "udip-dip; ddip; tracks", 100, -0.1, 0.1);
  h405 = fs->make<TH1D>("h405", "uz0-z0; dz0; tracks", 100, -0.1, 0.1);

  h406 = fs->make<TH1D>("h406", "valid tracker hits; tracker hits; tracks", 51, -0.5, 50.5);
  h407 = fs->make<TH1D>("h407", "valid pixel barrel hits; valid pixel barrel hits; tracks", 11, -0.5, 10.5);
  h408 = fs->make<TH1D>("h408", "tracker layers; tracker layers; tracks", 31, -0.5, 30.5);
  h409 = fs->make<TProfile>("h409", "angle of incidence; phi2 [deg]; phi inc 2 [deg]", 180, -180, 180, -90, 90);

  h410 = fs->make<TH1D>("h410", "residuals PXB2 dca2; dca2 [um]; hits", 100, -150, 150);
  h411 = fs->make<TH1D>("h411", "residuals PXB2 dz2 ; dz2  [um]; hits", 100, -300, 300);
  //
  // mean resid profiles:
  //
  h412 = fs->make<TProfile>("h412", "PXB2 dxy vs phi; phi2 [deg]; <dxy2> [um]", 180, -180, 180, -99, 99);
  h413 = fs->make<TProfile>("h413", "PXB2 dz  vs phi; phi2 [deg]; <dz2>  [um]", 180, -180, 180, -99, 199);

  h414 = fs->make<TProfile>("h414", "PXB2 dxy vs z; z2 [cm]; <dxy2> [um]", 80, -20, 20, -99, 99);
  h415 = fs->make<TProfile>("h415", "PXB2 dz  vs z; z2 [cm]; <dz2>  [um]", 80, -20, 20, -199, 199);

  h416 = fs->make<TProfile>("h416", "PXB2 dxy vs pt; log(pt [GeV]); <dxy2> [um]", 20, 0, 2, -99, 99);
  h417 = fs->make<TProfile>("h417", "PXB2 dz  vs pt; log(pt [GeV]); <dz2>  [um]", 20, 0, 2, -199, 199);

  h418 = fs->make<TProfile>("h418", "PXB2 dxy vs pt +; log(pt [GeV]); <dxy2> [um]", 20, 0, 2, -99, 99);
  h419 = fs->make<TProfile>("h419", "PXB2 dxy vs pt -; log(pt [GeV]); <dxy2> [um]", 20, 0, 2, -99, 99);

  h420 = fs->make<TH1D>("h420", "residuals PXB2 dca2, pt > 12; dca2 [um]; hits", 100, -150, 150);
  h421 = fs->make<TH1D>("h421", "residuals PXB2 dz2,  pt > 12; dz2  [um]; hits", 100, -300, 300);
  //
  // width profiles:
  //
  h422 = fs->make<TProfile>("h422", "PXB2 sxy vs phi; phi2 [deg]; sxy [um]", 360, -180, 180, 0, 99);
  h423 = fs->make<TProfile>("h423", "PXB2 sz  vs phi; phi2 [deg]; sz  [um]", 360, -180, 180, 0, 199);

  h424 = fs->make<TProfile>("h424", "PXB2 sxy vs z; z2 [cm]; sxy [um]", 80, -20, 20, 0, 99);
  h425 = fs->make<TProfile>("h425", "PXB2 sz  vs z; z2 [cm]; sz  [um]", 80, -20, 20, 0, 199);

  h426 = fs->make<TProfile>("h426", "PXB2 sxy vs pt; log(pt [GeV]); sxy [um]", 20, 0, 2, 0, 99);
  h427 = fs->make<TProfile>("h427", "PXB2 sz  vs pt; log(pt [GeV]); sz  [um]", 20, 0, 2, 0, 199);

  h428 = fs->make<TProfile>("h428", "PXB2 sxy vs dip; dip [deg]; sxy [um]", 70, -70, 70, 0, 99);
  h429 = fs->make<TProfile>("h429", "PXB2 sz  vs dip; dip [deg]; sz  [um]", 70, -70, 70, 0, 199);

  h430 = fs->make<TH1D>("h430", "residuals PXB2 dca2; dca2 [um]; hits", 100, -150, 150);
  h431 = fs->make<TH1D>("h431", "residuals PXB2 dz2;  dz2  [um]; hits", 100, -300, 300);

  h432 = fs->make<TH1D>("h432", "residuals PXB2 dz2, 18 < |dip| < 50; dz2 [um]; hits", 100, -300, 300);
  h433 = fs->make<TProfile>("h433", "PXB2 sz vs pt, best dip; log(pt [GeV]); sz [um]", 20, 0, 2, 0, 199);

  h434 = fs->make<TProfile>("h434", "PXB2 sxy vs inc; phi inc 2 [deg]; sxy [um]", 40, -10, 10, 0, 99);

  h435 = fs->make<TH1D>("h435", "ukap-kap; dkap; tracks", 100, -0.01, 0.01);
  h436 = fs->make<TH1D>("h436", "uphi-phi; dphi; tracks", 100, -0.1, 0.1);
  h437 = fs->make<TH1D>("h437", "udca-dca; ddca; tracks", 100, -0.1, 0.1);

  h440 = fs->make<TH1D>("h440", "pixel track dcap, pt > 2; dcap [um]; hits", 100, -1000, 1000);
  h441 = fs->make<TH1D>("h441", "pixel track dcap, pt > 4; dcap [um]; hits", 100, -1000, 1000);

  h442 = fs->make<TProfile>("h442", "pixel track  dcap vs phi; phi2 [deg]; <dcap> [um]", 180, -180, 180, -500, 499);
  h443 = fs->make<TProfile>("h443", "pixel tracks  dcap vs pt; log(pt [GeV]); <dcap> [um]", 20, 0, 2, -500, 499);

  h444 = fs->make<TProfile>("h444", "pixel track sdcap vs phi; phi2 [deg]; sdcap [um]", 180, -180, 180, 0, 499);
  h445 = fs->make<TProfile>("h445", "pixel tracks sdcap vs pt; log(pt [GeV]); sdcap [um]", 20, 0, 2, 0, 499);

  h450 = fs->make<TH1D>("h450", "residuals PXB2 dca2; dca2 [um]; hits", 100, -150, 150);
  h451 = fs->make<TH1D>("h451", "residuals PXB2 dz2;  dz2  [um]; hits", 100, -300, 300);
}
//
//----------------------------------------------------------------------
// method called for each event:
//
void Triplet::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo = iSetup.getHandle(trackerTopoToken_);

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace math;

  const double pi = 4 * atan(1);
  const double wt = 180 / pi;
  const double twopi = 2 * pi;
  const double pihalf = 2 * atan(1);
  //const double sqrtpihalf = sqrt(pihalf);

  myCounters::neve++;

  if (myCounters::prevrun != iEvent.run()) {
    time_t unixZeit = iEvent.time().unixTime();

    cout << "new run " << iEvent.run();
    cout << ", LumiBlock " << iEvent.luminosityBlock();
    cout << " taken " << ctime(&unixZeit);  // ctime has endline

    myCounters::prevrun = iEvent.run();

  }  // new run

  int idbg = 0;
  if (myCounters::neve < 1)
    idbg = 1;
  //
  //--------------------------------------------------------------------
  // beam spot:
  //
  edm::Handle<reco::BeamSpot> rbs;
  iEvent.getByToken(bsToken_, rbs);

  XYZPoint bsP = XYZPoint(0, 0, 0);
  //int ibs = 0;

  if (!rbs.failedToGet() && rbs.isValid()) {
    //ibs = 1;
    h000->Fill(rbs->betaStar());
    h001->Fill(rbs->emittanceX());
    h002->Fill(rbs->BeamWidthX() * 1e4);
    h003->Fill(rbs->BeamWidthY() * 1e4);
    h004->Fill(rbs->sigmaZ());
    h005->Fill(rbs->x0());
    h006->Fill(rbs->y0());
    h007->Fill(rbs->z0());
    h008->Fill(rbs->dxdz() * 1e3);
    h009->Fill(rbs->dydz() * 1e3);
    bsP = XYZPoint(rbs->x0(), rbs->y0(), rbs->z0());

    if (idbg) {
      cout << "beam spot x " << rbs->x0();
      cout << ", y " << rbs->y0();
      cout << ", z " << rbs->z0();
      cout << endl;
    }

  }  // bs valid
  //
  //--------------------------------------------------------------------
  // primary vertices:
  //
  Handle<VertexCollection> vertices;
  iEvent.getByToken(vtxToken_, vertices);

  if (vertices.failedToGet())
    return;
  if (!vertices.isValid())
    return;

  h010->Fill(vertices->size());

  // need vertex global point for tracks
  // from #include "DataFormats/GeometryVector/interface/GlobalPoint.h"
  // Global points are three-dimensional by default
  // typedef Global3DPoint  GlobalPoint;
  // typedef Point3DBase< float, GlobalTag> Global3DPoint;

  XYZPoint vtxN = XYZPoint(0, 0, 0);
  XYZPoint vtxP = XYZPoint(0, 0, 0);

  double bestNdof = 0;
  double maxSumPt = 0;
  Vertex bestPvx;

  for (VertexCollection::const_iterator iVertex = vertices->begin(); iVertex != vertices->end(); ++iVertex) {
    if (!iVertex->isValid())
      h011->Fill(iVertex->z());
    else {
      if (iVertex->isFake())
        h012->Fill(iVertex->z());

      else {
        h013->Fill(iVertex->z());
        h014->Fill(iVertex->x());
        h015->Fill(iVertex->y());
        h016->Fill(iVertex->ndof());
        h017->Fill(iVertex->ndof());

        if (idbg) {
          cout << "vertex";
          cout << ": x " << iVertex->x();
          cout << ", y " << iVertex->y();
          cout << ", z " << iVertex->z();
          cout << ", ndof " << iVertex->ndof();
          cout << ", sumpt " << iVertex->p4().pt();
          cout << endl;
        }

        if (iVertex->hasRefittedTracks())
          h018->Fill(iVertex->z());  // empty in Zmumu sample
        else
          h019->Fill(iVertex->z());  // all here: why?

        if (iVertex->ndof() > bestNdof) {
          bestNdof = iVertex->ndof();
          vtxN = XYZPoint(iVertex->x(), iVertex->y(), iVertex->z());
        }

        h021->Fill(iVertex->p4().pt());

        if (iVertex->p4().pt() > maxSumPt) {
          maxSumPt = iVertex->p4().pt();
          vtxP = XYZPoint(iVertex->x(), iVertex->y(), iVertex->z());
          bestPvx = *iVertex;
        }
      }  // non-fake
    }    //valid
  }      // loop over vertices

  h022->Fill(maxSumPt);

#ifndef SIM
  if (maxSumPt < 1)
    return;
#endif
  if (maxSumPt < 1)
    vtxP = vtxN;

  /*
  if( ibs ) {
    vtxP.SetX( bsP.x() ); // beam spot. should take tilt into account!
    vtxP.SetY( bsP.y() ); // beam spot. should take tilt into account!
  }
  */

  h023->Fill(vtxP.x());
  h024->Fill(vtxP.y());
  h025->Fill(vtxP.z());

  //double xbs = 0;
  //double ybs = 0;
  //if( ibs ) {
  //xbs = bsP.x();
  //ybs = bsP.y();
  //}
  //else {
  //xbs = vtxP.x();
  //ybs = vtxP.y();
  //}
  //
  //--------------------------------------------------------------------
  // MET:
  //
  //   edm::Handle< edm::View<reco::PFMET> > pfMEThandle;
  //   iEvent.getByLabel("pfMet", pfMEThandle);

  //   if( !pfMEThandle.failedToGet() && pfMEThandle.isValid()){

  //     h026->Fill( pfMEThandle->front().sumEt() );
  //     h027->Fill( pfMEThandle->front().et() );

  //   }
  //
  //--------------------------------------------------------------------
  // tracks:
  //
  Handle<TrackCollection> tracks;

  iEvent.getByToken(trackToken_, tracks);

  if (tracks.failedToGet())
    return;
  if (!tracks.isValid())
    return;

  h040->Fill(tracks->size());
  //
  // get tracker geometry:
  //
  edm::ESHandle<TrackerGeometry> pDD = iSetup.getHandle(trackerGeomToken_);

  if (!pDD.isValid()) {
    cout << "Unable to find TrackerDigiGeometry. Return\n";
    return;
  }
  //
  // loop over tracker detectors:
  //
  for (TrackerGeometry::DetContainer::const_iterator idet = pDD->dets().begin(); idet != pDD->dets().end(); ++idet) {
    DetId mydetId = (*idet)->geographicalId();
    uint32_t mysubDet = mydetId.subdetId();

    if (idbg) {
      cout << "Det " << mydetId.det();
      cout << ", subDet " << mydetId.subdetId();

      if (mysubDet == PixelSubdetector::PixelBarrel) {
        cout << ": PXB layer " << tTopo->pxbLayer(mydetId);
        cout << ", ladder " << tTopo->pxbLadder(mydetId);
        cout << ", module " << tTopo->pxbModule(mydetId);
        cout << ", at R " << (*idet)->position().perp();
        cout << ", F " << (*idet)->position().barePhi() * wt;
        cout << ", z " << (*idet)->position().z();
        cout << endl;
        cout << "rot x";
        cout << "\t" << (*idet)->rotation().xx();
        cout << "\t" << (*idet)->rotation().xy();
        cout << "\t" << (*idet)->rotation().xz();
        cout << endl;
        cout << "rot y";
        cout << "\t" << (*idet)->rotation().yx();
        cout << "\t" << (*idet)->rotation().yy();
        cout << "\t" << (*idet)->rotation().yz();
        cout << endl;
        cout << "rot z";
        cout << "\t" << (*idet)->rotation().zx();
        cout << "\t" << (*idet)->rotation().zy();
        cout << "\t" << (*idet)->rotation().zz();
        cout << endl;
        //
        // normal vector: includes alignment (varies from module to module along z on one ladder)
        // neighbouring ladders alternate with inward/outward orientation
        //
        cout << "normal";
        cout << ": x " << (*idet)->surface().normalVector().x();
        cout << ", y " << (*idet)->surface().normalVector().y();
        cout << ", z " << (*idet)->surface().normalVector().z();
        cout << ", f " << (*idet)->surface().normalVector().barePhi() * wt;

      }  //PXB

      if (mysubDet == PixelSubdetector::PixelEndcap) {
        cout << ": PXD side " << tTopo->pxfSide(mydetId);
        cout << ", disk " << tTopo->pxfDisk(mydetId);
        cout << ", blade " << tTopo->pxfBlade(mydetId);
        cout << ", panel " << tTopo->pxfPanel(mydetId);
        cout << ", module " << tTopo->pxfModule(mydetId);
        cout << ", at R " << (*idet)->position().perp();
        cout << ", F " << (*idet)->position().barePhi() * wt;
        cout << ", z " << (*idet)->position().z();
        cout << endl;
        cout << "rot x";
        cout << "\t" << (*idet)->rotation().xx();
        cout << "\t" << (*idet)->rotation().xy();
        cout << "\t" << (*idet)->rotation().xz();
        cout << endl;
        cout << "rot y";
        cout << "\t" << (*idet)->rotation().yx();
        cout << "\t" << (*idet)->rotation().yy();
        cout << "\t" << (*idet)->rotation().yz();
        cout << endl;
        cout << "rot z";
        cout << "\t" << (*idet)->rotation().zx();
        cout << "\t" << (*idet)->rotation().zy();
        cout << "\t" << (*idet)->rotation().zz();
        cout << endl;
        cout << "normal";
        cout << ": x " << (*idet)->surface().normalVector().x();
        cout << ", y " << (*idet)->surface().normalVector().y();
        cout << ", z " << (*idet)->surface().normalVector().z();
        cout << ", f " << (*idet)->surface().normalVector().barePhi() * wt;

      }  //PXD

      cout << endl;

    }  //idbg

  }  //idet

  //
  // transient track builder, needs B-field from data base (global tag in .py)
  //
  edm::ESHandle<TransientTrackBuilder> theB = iSetup.getHandle(transientTrackBuilderToken_);
  //
  // transient rec hits:
  //
  edm::ESHandle<TransientTrackingRecHitBuilder> hitBuilder = iSetup.getHandle(transientTrackingRecHitBuilderToken_);
  //
  //
  //
  double sumpt = 0;
  double sumq = 0;
  int kk = -1;
  Surface::GlobalPoint origin = Surface::GlobalPoint(0, 0, 0);
  //
  //----------------------------------------------------------------------------
  // Tracks:
  //
  for (TrackCollection::const_iterator iTrack = tracks->begin(); iTrack != tracks->end(); ++iTrack) {
    kk++;

    // cpt = cqRB = 0.3*R[m]*B[T] = 1.14*R[m] for B=3.8T
    // D = 2R = 2*pt/1.14
    // calo: D = 1.3 m => pt = 0.74 GeV/c

    double pt = iTrack->pt();

    if (pt < 0.75)
      continue;  // curls up
    //if( pt < 1.75 ) continue;// want sharper image

    if (abs(iTrack->dxy(vtxP)) > 5 * iTrack->dxyError())
      continue;  // not prompt

    double logpt = log(pt) / log(10);
    double charge = iTrack->charge();
    h041->Fill(charge);
    h042->Fill(pt);
    h043->Fill(pt);

    if (idbg) {
      cout << "Track " << kk;
      cout << ": pt " << iTrack->pt();
      cout << ", eta " << iTrack->eta();
      cout << ", phi " << iTrack->phi() * wt;
      cout << ", dxyv " << iTrack->dxy(vtxP);
      cout << ", dzv " << iTrack->dz(vtxP);
      cout << endl;
    }

    const reco::HitPattern &hp = iTrack->hitPattern();

    h045->Fill(hp.numberOfValidTrackerHits());
    h046->Fill(hp.numberOfValidPixelBarrelHits());
    h047->Fill(hp.trackerLayersWithMeasurement());
    h048->Fill(hp.pixelBarrelLayersWithMeasurement());

    double phi = iTrack->phi();
    double dca = iTrack->d0();  // w.r.t. origin
    //double dca = -iTrack->dxy(); // dxy = -d0
    double dip = iTrack->lambda();
    double z0 = iTrack->dz();
    double tet = pihalf - dip;
    //double eta = iTrack->eta();
    //
    // transient track:
    //
    TransientTrack tTrack = theB->build(*iTrack);

    double kap = tTrack.initialFreeState().transverseCurvature();

    TrajectoryStateClosestToPoint tscp = tTrack.trajectoryStateClosestToPoint(origin);

    //cout<<pt<<" "<<kap<<" ";

    if (tscp.isValid()) {
      h057->Fill(tscp.referencePoint().x());  // 0.0
      h058->Fill(tscp.referencePoint().y());  // 0.0
      h059->Fill(tscp.referencePoint().z());  // 0.0
      kap = tscp.perigeeParameters().transverseCurvature();
      phi = tscp.perigeeParameters().phi();
      dca = tscp.perigeeParameters().transverseImpactParameter();
      tet = tscp.perigeeParameters().theta();
      z0 = tscp.perigeeParameters().longitudinalImpactParameter();
      dip = pihalf - tet;

      h051->Fill(kap - tTrack.initialFreeState().transverseCurvature());
      h052->Fill(phi - iTrack->phi());
      h053->Fill(dca - iTrack->d0());
      h054->Fill(dip - iTrack->lambda());
      h055->Fill(z0 - iTrack->dz());
    }

    double tmp = abs(kap * pt);
    h056->Fill(tmp);
    double rho1 = pt / 0.0114257;
    if (charge > 0)
      rho1 = -rho1;
    double kap1 = 1. / rho1;
    double tmp1 = (kap1 - kap);
    h049->Fill(tmp1);

    //cout<<pt<<" "<<kap<<" "<<tmp<<" "<<charge<<" "<<kap1<<endl;

    double cf = cos(phi);
    double sf = sin(phi);
    //double xdca =  dca * sf;
    //double ydca = -dca * cf;

    //double tt = tan(tet);

    //double rinv = -kap; // Karimaki
    //double rho = 1/kap;
    double erd = 1.0 - kap * dca;
    double drd = dca * (0.5 * kap * dca - 1.0);  // 0.5 * kap * dca**2 - dca;
    double hkk = 0.5 * kap * kap;
    //
    // track w.r.t. beam (cirmov):
    //
    //double dp = -xbs*sf + ybs*cf + dca;
    //double dl = -xbs*cf - ybs*sf;
    //double sa = 2*dp + rinv*(dp*dp+dl*dl);
    //double dcap = sa / ( 1 + sqrt(1 + rinv*sa) );// distance to beam
    //double ud = 1 + rinv*dca;
    //double phip = atan2( -rinv*xbs + ud*sf, rinv*ybs + ud*cf );//direction
    //
    // track at R(PXB1), from FUNPHI, FUNLEN:
    //
    double R1 = 4.4;  // PXB1

    double s1 = 0;
    double fpos1 = phi - pihalf;

    if (R1 >= abs(dca)) {
      //
      // sin(delta phi):
      //
      double sindp = (0.5 * kap * (R1 * R1 + dca * dca) - dca) / (R1 * erd);
      fpos1 = phi + asin(sindp);  // phi position at R1
      //
      // sin(alpha):
      //
      double sina = R1 * kap * sqrt(1.0 - sindp * sindp);
      //
      // s = alpha / kappa:
      //
      if (sina >= 1.0)
        s1 = pi / kap;
      else {
        if (sina <= -1.0)
          s1 = -pi / kap;
        else
          s1 = asin(sina) / kap;  //always positive
      }
      //
      // Check direction: limit to half-turn
      //
      if (hkk * (R1 * R1 - dca * dca) > erd)
        s1 = pi / abs(kap) - s1;  // always positive

    }  // R1 > dca

    if (fpos1 > pi)
      fpos1 -= twopi;
    else if (fpos1 < -pi)
      fpos1 += twopi;

    double zR1 = z0 + s1 * tan(dip);  // z at R1
    //
    //--------------------------------------------------------------------------
    // loop over tracker detectors:
    //
    double xcrss[99];
    double ycrss[99];
    double zcrss[99];
    int ncrss = 0;

    for (TrackerGeometry::DetContainer::const_iterator idet = pDD->dets().begin(); idet != pDD->dets().end(); ++idet) {
      DetId mydetId = (*idet)->geographicalId();
      uint32_t mysubDet = mydetId.subdetId();

      if (mysubDet != PixelSubdetector::PixelBarrel)
        continue;
      /*
	cout << ": PXB layer " << tTopo->pxbLayer(mydetId);
	cout << ", ladder " << tTopo->pxbLadder(mydetId);
	cout << ", module " << tTopo->pxbModule(mydetId);
	cout << ", at R1 " << (*idet)->position().perp();
	cout << ", F " << (*idet)->position().barePhi()*wt;
	cout << ", z " << (*idet)->position().z();
	cout << endl;
      */

      if (tTopo->pxbLayer(mydetId) == 1) {
        double dz = zR1 - (*idet)->position().z();

        if (abs(dz) > 4.0)
          continue;

        double df = fpos1 - (*idet)->position().barePhi();

        if (df > pi)
          df -= twopi;
        else if (df < -pi)
          df += twopi;

        if (abs(df) * wt > 36.0)
          continue;
        //
        // normal vector: includes alignment (varies from module to module along z on one ladder)
        // neighbouring ladders alternate with inward/outward orientation
        //
        /*
	  cout << "normal";
	  cout << ": x " << (*idet)->surface().normalVector().x();
	  cout << ", y " << (*idet)->surface().normalVector().y();
	  cout << ", z " << (*idet)->surface().normalVector().z();
	  cout << ", f " << (*idet)->surface().normalVector().barePhi()*wt;
	  cout << endl;
	*/

        double phiN = (*idet)->surface().normalVector().barePhi();  //normal vector

        double phidet = phiN - pihalf;  // orientation of sensor plane in x-y
        double ux = cos(phidet);        // vector in sensor plane
        double uy = sin(phidet);
        double x = (*idet)->position().x();
        double y = (*idet)->position().y();
        //
        // intersect helix with line: FUNRXY (in FUNPHI) from V. Blobel
        // factor f for intersect point (x + f*ux, y + f*uy)
        //
        double a = kap * (ux * ux + uy * uy) * 0.5;
        double b = erd * (ux * sf - uy * cf) + kap * (ux * x + uy * y);
        double c = drd + erd * (x * sf - y * cf) + kap * (x * x + y * y) * 0.5;
        double dis = b * b - 4.0 * a * c;
        double f = 0;

        if (dis > 0) {
          dis = sqrt(dis);
          double f1 = 0;
          double f2 = 0;

          if (b < 0) {
            f1 = 0.5 * (dis - b) / a;
            f2 = 2.0 * c / (dis - b);
          } else {
            f1 = -0.5 * (dis + b) / a;
            f2 = -2.0 * c / (dis + b);
          }

          f = f1;
          if (abs(f2) < abs(f1))
            f = f2;

        }  //dis

        xcrss[ncrss] = x + f * ux;
        ycrss[ncrss] = y + f * uy;
        double r = sqrt(xcrss[ncrss] * xcrss[ncrss] + ycrss[ncrss] * ycrss[ncrss]);

        double s = 0;

        if (r >= abs(dca)) {
          double sindp = (0.5 * kap * (r * r + dca * dca) - dca) / (r * erd);
          double sina = r * kap * sqrt(1.0 - sindp * sindp);
          if (sina >= 1.0)
            s = pi / kap;
          else {
            if (sina <= -1.0)
              s = -pi / kap;
            else
              s = asin(sina) / kap;
          }
          if (hkk * (r * r - dca * dca) > erd)
            s = pi / abs(kap) - s;
        }

        zcrss[ncrss] = z0 + s * tan(dip);  // z at r

        ncrss++;

      }  //PXB1

    }  //idet
    //
    //--------------------------------------------------------------------------
    // rec hits from track extra:
    //
    if (iTrack->extra().isNonnull() && iTrack->extra().isAvailable()) {
      h044->Fill(tTrack.recHitsSize());  // tTrack

      double x1 = 0;
      double y1 = 0;
      double z1 = 0;
      double x2 = 0;
      double y2 = 0;
      double z2 = 0;
      double x3 = 0;
      double y3 = 0;
      double z3 = 0;
      int n1 = 0;
      int n2 = 0;
      int n3 = 0;
      //double phiN2 = 0;

      for (trackingRecHit_iterator irecHit = iTrack->recHitsBegin(); irecHit != iTrack->recHitsEnd(); ++irecHit) {
        if ((*irecHit)->isValid()) {
          DetId detId = (*irecHit)->geographicalId();

          // enum Detector { Tracker=1, Muon=2, Ecal=3, Hcal=4, Calo=5 };

          if (detId.det() != 1) {
            cout << "rec hit ID = " << detId.det() << " not in tracker!?!?\n";
            continue;
          }

          uint32_t subDet = detId.subdetId();

          // enum SubDetector{ PixelBarrel=1, PixelEndcap=2 };
          // enum SubDetector{ TIB=3, TID=4, TOB=5, TEC=6 };

          h060->Fill(subDet);
          //
          // build hit: (from what?)
          //
          TransientTrackingRecHit::RecHitPointer trecHit = hitBuilder->build(&*(*irecHit));

          double xloc = trecHit->localPosition().x();  // 1st meas coord
          double yloc = trecHit->localPosition().y();  // 2nd meas coord or zero
          //double zloc = trecHit->localPosition().z();// up, always zero
          h061->Fill(xloc);
          h062->Fill(yloc);

          double gX = trecHit->globalPosition().x();
          double gY = trecHit->globalPosition().y();
          double gZ = trecHit->globalPosition().z();
          double gF = atan2(gY, gX);
          double gR = sqrt(gX * gX + gY * gY);

          h063->Fill(gR);
          h064->Fill(gF * wt);
          h065->Fill(gZ);

          //	  GeomDet* igeomdet = trecHit->det();
          //double phiN = trecHit->det()->surface().normalVector().barePhi();//normal vector

          if (subDet == PixelSubdetector::PixelBarrel || subDet == StripSubdetector::TIB ||
              subDet == StripSubdetector::TOB) {  // barrel

            h066->Fill(gX, gY);

          }  //barrel

          if (subDet == PixelSubdetector::PixelBarrel) {
            int ilay = tTopo->pxbLayer(detId);
            int ilad = tTopo->pxbLadder(detId);
            int imod = tTopo->pxbModule(detId);
            bool halfmod = 0;

            h100->Fill(ilay);  // 1,2,3

            if (ilay == 1) {
              n1++;
              x1 = gX;
              y1 = gY;
              z1 = gZ;

              h101->Fill(ilad);  // 1..20
              h102->Fill(imod);  // 1..8

              h103->Fill(gR);
              h104->Fill(gF * wt);
              h105->Fill(gZ);

              h106->Fill(gF * wt, gZ);  // phi-z of hit

              if (ilad == 5)
                halfmod = 1;
              else if (ilad == 6)
                halfmod = 1;
              else if (ilad == 15)
                halfmod = 1;
              else if (ilad == 16)
                halfmod = 1;

              if (!halfmod) {
                h107->Fill(xloc, yloc);  // hit within one module
              }
              //
              // my crossings:
              //
              for (int icrss = 0; icrss < ncrss; ++icrss) {
                double fcrss = atan2(ycrss[icrss], xcrss[icrss]);
                double df = gF - fcrss;
                if (df > pi)
                  df -= twopi;
                else if (df < -pi)
                  df += twopi;
                double du = gR * df;
                double dz = gZ - zcrss[icrss];

                if (abs(du) < 0.01 && abs(dz) < 0.05)
                  h111->Fill(gX, gY);
                h112->Fill(du * 1E4);
                h113->Fill(dz * 1E4);

                if (abs(dz) < 0.02)
                  h114->Fill(du * 1E4);
                if (abs(du) < 0.01)
                  h115->Fill(dz * 1E4);

              }  //crss

            }  //PXB1

            if (ilay == 2) {
              n2++;
              x2 = gX;
              y2 = gY;
              z2 = gZ;
              //phiN2 = phiN;

              h201->Fill(ilad);  // 1..32
              h202->Fill(imod);  //1..8

              h203->Fill(gR);
              h204->Fill(gF * wt);
              h205->Fill(gZ);

              h206->Fill(gF * wt, gZ);  // phi-z of hit

              if (ilad == 8)
                halfmod = 1;
              else if (ilad == 9)
                halfmod = 1;
              else if (ilad == 24)
                halfmod = 1;
              else if (ilad == 25)
                halfmod = 1;

              if (!halfmod) {
                h207->Fill(xloc, yloc);  // hit within one module
              }

            }  //PXB2

            if (ilay == 3) {
              n3++;
              x3 = gX;
              y3 = gY;
              z3 = gZ;

              h301->Fill(ilad);  //1..44
              h302->Fill(imod);  //1..8

              h303->Fill(gR);
              h304->Fill(gF * wt);
              h305->Fill(gZ);

              h306->Fill(gF * wt, gZ);  // phi-z of hit

              if (ilad == 11)
                halfmod = 1;
              if (ilad == 12)
                halfmod = 1;
              if (ilad == 33)
                halfmod = 1;
              if (ilad == 34)
                halfmod = 1;

              if (!halfmod) {
                h307->Fill(xloc, yloc);  // hit within one module
              }

            }  //PXB3

          }  //PXB

        }  //valid

      }  //loop rechits
      //
      // 1-2-3 triplet:
      //
      //if( hp.pixelBarrelLayersWithMeasurement() == 3 ){
      if (n1 * n2 * n3 > 0) {
        double dca2 = 0., dz2 = 0.;
        //triplets(x1,y1,z1,x2,y2,z2,x3,y3,z3,kap,dca2,dz2);
        double ptsig = pt;
        if (charge < 0.)
          ptsig = -pt;
        triplets(x1, y1, z1, x2, y2, z2, x3, y3, z3, ptsig, dca2, dz2);

        if (pt > 4) {
          h410->Fill(dca2 * 1E4);
          h411->Fill(dz2 * 1E4);
        }
        if (pt > 12) {
          h420->Fill(dca2 * 1E4);
          h421->Fill(dz2 * 1E4);
          if (hp.trackerLayersWithMeasurement() > 8) {
            h430->Fill(dca2 * 1E4);
            h431->Fill(dz2 * 1E4);
          }
          //if( phiinc*wt > -1 && phiinc*wt < 7 ){
          //h450->Fill( dca2*1E4 );
          //h451->Fill( dz2*1E4 );
          //}
        }

        //
        // residual profiles: alignment check
        //
        if (pt > 4) {
          //h412->Fill( f2*wt, dca2*1E4 );
          //h413->Fill( f2*wt, dz2*1E4 );

          h414->Fill(z2, dca2 * 1E4);
          h415->Fill(z2, dz2 * 1E4);
        }

        h416->Fill(logpt, dca2 * 1E4);
        h417->Fill(logpt, dz2 * 1E4);
        if (iTrack->charge() > 0)
          h418->Fill(logpt, dca2 * 1E4);
        else
          h419->Fill(logpt, dca2 * 1E4);

        // profile of abs(dca) gives mean abs(dca):
        // mean of abs(Gauss) = 0.7979 * RMS = 1/sqrt(pi/2)
        // => rms = sqrt(pi/2) * mean of abs (sqrt(pi/2) = 1.2533)
        // point resolution = 1/sqrt(3/2) * triplet middle residual width
        // => sqrt(pi/2)*sqrt(2/3) = sqrt(pi/3) = 1.0233, almost one

        if (pt > 4) {
          //h422->Fill( f2*wt, abs(dca2)*1E4 );
          //h423->Fill( f2*wt, abs(dz2)*1E4 );

          h424->Fill(z2, abs(dca2) * 1E4);
          h425->Fill(z2, abs(dz2) * 1E4);

          //h428->Fill( udip*wt, abs(dca2)*1E4 );
          //h429->Fill( udip*wt, abs(dz2)*1E4 );
          //if( abs(dip)*wt > 18 && abs(dip)*wt < 50 ) h432->Fill( dz2*1E4 );

          //h434->Fill( phiinc*wt, abs(dca2)*1E4 );

        }  //pt

        h426->Fill(logpt, abs(dca2) * 1E4);
        h427->Fill(logpt, abs(dz2) * 1E4);
        //if( abs(dip)*wt > 18 && abs(dip)*wt < 50 ) h433->Fill( logpt, abs(dz2)*1E4 );

      }  //3 PXB layers

    }  //extra

    sumpt += iTrack->pt();
    sumq += iTrack->charge();

  }  // loop over tracks

  h028->Fill(sumpt);
  h029->Fill(sumq);
}

void Triplet::triplets(double x1,
                       double y1,
                       double z1,
                       double x2,
                       double y2,
                       double z2,
                       double x3,
                       double y3,
                       double z3,
                       double ptsig,
                       double &dca2,
                       double &dz2) {
  using namespace std;
  const double pi = 4 * atan(1);
  //const double wt = 180/pi;
  const double twopi = 2 * pi;
  //const double pihalf = 2*atan(1);
  //const double sqrtpihalf = sqrt(pihalf);

  double pt = abs(ptsig);
  //double rho = pt/0.0114257;
  double kap = 0.0114257 / pt;
  if (ptsig > 0)
    kap = -kap;  // kap i snegative for positive charge

  double rho = 1 / kap;
  double rinv = -kap;  // Karimaki

  //double f2 = atan2( y2, x2 );//position angle

  //h406->Fill( hp.numberOfValidTrackerHits() );
  //h407->Fill( hp.numberOfValidPixelBarrelHits() );
  //h408->Fill( hp.trackerLayersWithMeasurement() );

  // Author: Johannes Gassner (15.11.1996)
  // Make track from 2 space points and kappa (cmz98/ftn/csmktr.f)
  // Definition of the Helix :
  //
  // x( t ) = X0 + KAPPA^-1 * SIN( PHI0 + t )
  // y( t ) = Y0 - KAPPA^-1 * COS( PHI0 + t )          t > 0
  // z( t ) = Z0 + KAPPA^-1 * TAN( DIP ) * t

  // Center of the helix in the xy-projection:

  // X0 = + ( DCA - KAPPA^-1 ) * SIN( PHI0 )
  // Y0 = - ( DCA - KAPPA^-1 ) * COS( PHI0 )
  //
  // Point 1 must be in the inner layer, 3 in the outer:
  //
  double r1 = sqrt(x1 * x1 + y1 * y1);
  double r3 = sqrt(x3 * x3 + y3 * y3);

  if (r3 - r1 < 2.0)
    cout << "warn r1 = " << r1 << ", r3 = " << r3 << endl;
  //
  // Calculate the centre of the helix in xy-projection that
  // transverses the two spacepoints. The points with the same
  // distance from the two points are lying on a line.
  // LAMBDA is the distance between the point in the middle of
  // the two spacepoints and the centre of the helix.
  //
  // we already have kap and rho = 1/kap
  //

  double lam = sqrt(-0.25 + rho * rho / ((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3)));
  //
  // There are two solutions, the sign of kap gives the information
  // which of them is correct.
  //
  if (kap > 0)
    lam = -lam;
  //
  // ( X0, Y0 ) is the centre of the circle that describes the helix in xy-projection.
  //
  double x0 = 0.5 * (x1 + x3) + lam * (-y1 + y3);
  double y0 = 0.5 * (y1 + y3) + lam * (x1 - x3);
  //
  // Calculate theta :
  //
  double num = (y3 - y0) * (x1 - x0) - (x3 - x0) * (y1 - y0);
  double den = (x1 - x0) * (x3 - x0) + (y1 - y0) * (y3 - y0);
  double tandip = kap * (z3 - z1) / atan(num / den);
  //double udip = atan(tandip);
  //double utet = pihalf - udip;
  //
  // To get phi0 in the right intervall one must differ two cases
  // with positve and negative kap:
  //
  double uphi;
  if (kap > 0)
    uphi = atan2(-x0, y0);
  else
    uphi = atan2(x0, -y0);
  //
  // The distance of the closest approach DCA depends on the sign
  // of kappa :
  //
  double udca;
  if (kap > 0)
    udca = rho - sqrt(x0 * x0 + y0 * y0);
  else
    udca = rho + sqrt(x0 * x0 + y0 * y0);
  //
  // angle from first hit to dca point:
  //
  double dphi = atan(((x1 - x0) * y0 - (y1 - y0) * x0) / ((x1 - x0) * x0 + (y1 - y0) * y0));

  double uz0 = z1 + tandip * dphi * rho;

  //h401->Fill( z2 );
  //h402->Fill( uphi - iTrack->phi() );
  //h403->Fill( udca - iTrack->d0() );
  //h404->Fill( udip - iTrack->lambda() );
  //h405->Fill( uz0  - iTrack->dz() );
  //
  // interpolate to middle hit:
  // cirmov
  // we already have rinv = -kap
  //
  double cosphi = cos(uphi);
  double sinphi = sin(uphi);
  double dp = -x2 * sinphi + y2 * cosphi + udca;
  double dl = -x2 * cosphi - y2 * sinphi;
  double sa = 2 * dp + rinv * (dp * dp + dl * dl);
  dca2 = sa / (1 + sqrt(1 + rinv * sa));  // distance to hit 2

  double ud = 1 + rinv * udca;
  double phi2 = atan2(-rinv * x2 + ud * sinphi, rinv * y2 + ud * cosphi);  //direction

  //double phiinc = phi2 - phiN2;//angle of incidence in rphi w.r.t. normal vector
  //
  // phiN alternates inward/outward
  // reduce phiinc
  //if( phiinc > pihalf ) phiinc -= pi;
  //else if( phiinc < -pihalf ) phiinc += pi;
  //h409->Fill( f2*wt, phiinc*wt );
  //
  // arc length:
  //
  double xx = x2 + dca2 * sin(phi2);  // point on track
  double yy = y2 - dca2 * cos(phi2);

  double f0 = uphi;  //
  double kx = kap * xx;
  double ky = kap * yy;
  double kd = kap * udca;
  //
  // Solve track equation for s:
  //
  double dx = kx - (kd - 1) * sin(f0);
  double dy = ky + (kd - 1) * cos(f0);
  double ks = atan2(dx, -dy) - f0;  // turn angle
  //
  //---  Limit to half-turn:
  //
  if (ks > pi)
    ks = ks - twopi;
  else if (ks < -pi)
    ks = ks + twopi;

  double s = ks * rho;            // signed
  double uz2 = uz0 + s * tandip;  //track z at R2
  dz2 = z2 - uz2;
}
//----------------------------------------------------------------------
// method called just after ending the event loop:
//
void Triplet::endJob() { std::cout << "end of job after " << myCounters::neve << " events.\n"; }

//define this as a plug-in
DEFINE_FWK_MODULE(Triplet);
