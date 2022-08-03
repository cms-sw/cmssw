// -*- C++ -*-
//
// Package:    GlobalTrackerMuonAlignment
// Class:      GlobalTrackerMuonAlignment
//
/**\class GlobalTrackerMuonAlignment GlobalTrackerMuonAlignment.cc
 Alignment/GlobalTrackerMuonAlignment/src/GlobalTrackerMuonAlignment.cc

 Description: Producer of relative tracker and muon system alignment

 Implementation:
 A sample of global muons is used for the aligning tracker and muon system
 relatively as "rigid bodies", i.e. determining offset and rotation (6 numbers)

*/
//
// Original Author:  Alexandre Spiridonov
//         Created:  Fri Oct 16 15:59:05 CEST 2009
//
// $Id: GlobalTrackerMuonAlignment.cc,v 1.11 2012/07/16 12:17:53 eulisse Exp $
//

#include "Alignment/CommonAlignmentProducer/interface/AlignmentProducerBase.h"

// system include files
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

// user include files

// Framework
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

// references
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirectionChooser.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/SmartPropagator.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

// Database
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Alignment
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"

// Refit
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EventSetup.h" //MK Library

using namespace edm;
using namespace std;
using namespace reco;

//
// class declaration
//

class GlobalTrackerMuonAlignment : public edm::one::EDAnalyzer<> {
public:
  explicit GlobalTrackerMuonAlignment(const edm::ParameterSet &);
  ~GlobalTrackerMuonAlignment() override;
  void analyzeTrackTrajectory(const edm::Event &, const edm::EventSetup &);
  void bookHist();
  void fitHist();

  void gradientGlobal(GlobalVector &, GlobalVector &, GlobalVector &,
                      GlobalVector &, GlobalVector &, AlgebraicSymMatrix66 &);
  void gradientLocal(GlobalVector &, GlobalVector &, GlobalVector &,
                     GlobalVector &, GlobalVector &, CLHEP::HepSymMatrix &,
                     CLHEP::HepMatrix &, CLHEP::HepVector &,
                     AlgebraicVector4 &);
  void gradientGlobalAlg(GlobalVector &, GlobalVector &, GlobalVector &,
                         GlobalVector &, AlgebraicSymMatrix66 &);
  void misalignMuon(GlobalVector &, GlobalVector &, GlobalVector &,
                    GlobalVector &, GlobalVector &, GlobalVector &);
  void misalignMuonL(GlobalVector &, GlobalVector &, GlobalVector &,
                     GlobalVector &, GlobalVector &, GlobalVector &,
                     AlgebraicVector4 &, TrajectoryStateOnSurface &,
                     TrajectoryStateOnSurface &);
  void writeGlPosRcd(CLHEP::HepVector &d3);
  inline double CLHEP_dot(const CLHEP::HepVector &a,
                          const CLHEP::HepVector &b) {
    return a(1) * b(1) + a(2) * b(2) + a(3) * b(3);
  }

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>
      m_TkGeometryToken;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_MagFieldToken;
  edm::ESGetToken<Alignments, GlobalPositionRcd> m_globalPosToken;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> m_propToken;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord>
      m_ttrhBuilderToken;

  edm::InputTag ref_trackTags_; // Track Refitter
  edm::InputTag
      trackTags_; // used to select what tracks to read from configuration file

  edm::InputTag smuonTags_; // used to select what selected muons

  edm::EDGetTokenT<TrajTrackAssociationCollection> ref_track_;
  edm::EDGetTokenT<reco::MuonCollection> smuons_;

  string propagator_; // name of the propagator
  bool refitTrack_;   //                tracker track
  string rootOutFile_;
  string txtOutFile_;
  bool writeDB_; // write results to DB

  edm::ESWatcher<GlobalTrackingGeometryRecord> watchTrackingGeometry_;
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  const GlobalTrackingGeometry *trackingGeometry_;

  edm::ESWatcher<IdealMagneticFieldRecord> watchMagneticFieldRecord_;
  const MagneticField *magneticField_;

  edm::ESWatcher<TrackingComponentsRecord> watchTrackingComponentsRecord_;

  edm::ESWatcher<GlobalPositionRcd> watchGlobalPositionRcd_;
  const Alignments *globalPositionRcd_;

  KFTrajectoryFitter *theFitter;
  KFTrajectorySmoother *theSmoother;
  KFTrajectoryFitter *theFitterOp;
  KFTrajectorySmoother *theSmootherOp;
  bool defineFitter;
  MuonTransientTrackingRecHitBuilder *MuRHBuilder;
  const TransientTrackingRecHitBuilder *TTRHBuilder;

  //                            // LSF for global d(3):    Inf * d = Gfr
  AlgebraicVector3 Gfr;     // free terms
  AlgebraicSymMatrix33 Inf; // information matrix
  //                            // LSF for global d(6):    Hess * d = Grad
  CLHEP::HepVector
      Grad; // gradient of the objective function in global parameters
  CLHEP::HepMatrix Hess; // Hessian                  -- // ---

  CLHEP::HepVector
      GradL; // gradient of the objective function in local parameters
  CLHEP::HepMatrix HessL; // Hessian                  -- // ---

  int N_event; // processed events
  int N_track; // selected tracks

  std::vector<AlignTransform>::const_iterator
      iteratorTrackerRcd; // location of Tracker in container
                          // globalPositionRcd_->m_aligm
  std::vector<AlignTransform>::const_iterator
      iteratorMuonRcd; //              Muon
  std::vector<AlignTransform>::const_iterator
      iteratorEcalRcd; //              Ecal
  std::vector<AlignTransform>::const_iterator
      iteratorHcalRcd; //              Hcal

  CLHEP::HepVector MuGlShift; // evaluated global muon shifts
  CLHEP::HepVector MuGlAngle; // evaluated global muon angles

  std::string MuSelect; // what part of muon system is selected for 1st hit

  ofstream OutGlobalTxt; // output the vector of global alignment as text

  TFile *file;
  TH1F *histo;
  TH1F *histo2; // outerP
  TH1F *histo3; // outerLambda = PI/2-outerTheta
  TH1F *histo4; // phi
  TH1F *histo5; // outerR
  TH1F *histo6; // outerZ
  TH1F *histo7; // s
  TH1F *histo8; // chi^2 of muon-track

  TH1F *histo11; // |Rm-Rt|
  TH1F *histo12; // Xm-Xt
  TH1F *histo13; // Ym-Yt
  TH1F *histo14; // Zm-Zt
  TH1F *histo15; // Nxm-Nxt
  TH1F *histo16; // Nym-Nyt
  TH1F *histo17; // Nzm-Nzt
  TH1F *histo18; // Error X of inner track
  TH1F *histo19; // Error X of muon
  TH1F *histo20; // Error of Xm-Xt
  TH1F *histo21; // pull(Xm-Xt)
  TH1F *histo22; // pull(Ym-Yt)
  TH1F *histo23; // pull(Zm-Zt)
  TH1F *histo24; // pull(PXm-PXt)
  TH1F *histo25; // pull(PYm-Pyt)
  TH1F *histo26; // pull(PZm-PZt)
  TH1F *histo27; // Nx of tangent plane
  TH1F *histo28; // Ny of tangent plane
  TH1F *histo29; // lenght of inner track
  TH1F *histo30; // lenght of muon track
  TH1F *histo31; // chi2 local
  TH1F *histo32; // pull(Pxm/Pzm - Pxt/Pzt) local
  TH1F *histo33; // pull(Pym/Pzm - Pyt/Pzt) local
  TH1F *histo34; // pull(Xm - Xt) local
  TH1F *histo35; // pull(Ym - Yt) local

  TH2F *histo101; // Rtrack/muon vs Ztrack/muon
  TH2F *histo102; // Ytrack/muon vs Xtrack/muon
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
GlobalTrackerMuonAlignment::GlobalTrackerMuonAlignment(
    const edm::ParameterSet &iConfig)
    : m_TkGeometryToken(esConsumes()), m_MagFieldToken(esConsumes()),
      m_globalPosToken(esConsumes()),
      m_propToken(esConsumes(
          edm::ESInputTag("", iConfig.getParameter<string>("Propagator")))),
      m_ttrhBuilderToken(esConsumes(edm::ESInputTag("", "WithTrackAngle"))),
      ref_trackTags_(iConfig.getParameter<edm::InputTag>("ref_track")),
      smuonTags_(iConfig.getParameter<edm::InputTag>("smuons")),

      ref_track_(consumes<TrajTrackAssociationCollection>(
          iConfig.getParameter<InputTag>("ref_track"))),
      smuons_(consumes<reco::MuonCollection>(
          iConfig.getParameter<InputTag>("smuons"))),
      refitTrack_(iConfig.getParameter<bool>("refittrack")),
      rootOutFile_(iConfig.getUntrackedParameter<string>("rootOutFile")),
      txtOutFile_(iConfig.getUntrackedParameter<string>("txtOutFile")),
      writeDB_(iConfig.getUntrackedParameter<bool>("writeDB")), defineFitter(true) {}

GlobalTrackerMuonAlignment::~GlobalTrackerMuonAlignment() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void GlobalTrackerMuonAlignment::analyze(const edm::Event &iEvent,
                                         const edm::EventSetup &iSetup) {
  GlobalTrackerMuonAlignment::analyzeTrackTrajectory(iEvent, iSetup);
}

// ------------ method called once for job just before starting event loop
// ------------
void GlobalTrackerMuonAlignment::beginJob() {
  std::cout << "beginJob start" << std::endl;
  N_event = 0;
  N_track = 0;

  for (int i = 0; i <= 2; i++) {
    Gfr(i) = 0.;
    for (int j = 0; j <= 2; j++) {
      Inf(i, j) = 0.;
    }
  }

  Grad = CLHEP::HepVector(6, 0);
  Hess = CLHEP::HepSymMatrix(6, 0);

  GradL = CLHEP::HepVector(6, 0);
  HessL = CLHEP::HepSymMatrix(6, 0);

  // histograms
  TDirectory *dirsave = gDirectory;

  file = new TFile(rootOutFile_.c_str(), "recreate");
  const bool oldAddDir = TH1::AddDirectoryStatus();

  TH1::AddDirectory(true);

  this->bookHist();

  TH1::AddDirectory(oldAddDir);
  dirsave->cd();
}

// ------------ method called once each job just after ending the event loop
// ------------
void GlobalTrackerMuonAlignment::endJob() {

  this->fitHist();

  AlgebraicVector3 d(0., 0., 0.); // ------------  alignmnet  Global Algebraic

  AlgebraicSymMatrix33 InfI; // inverse it
  for (int i = 0; i <= 2; i++)
    for (int j = 0; j <= 2; j++) {
      if (j < i)
        continue;
      InfI(i, j) += Inf(i, j);
    }

  for (int i = 0; i <= 2; i++)
    for (int k = 0; k <= 2; k++)
      d(i) -= InfI(i, k) * Gfr(k);
  // end of Global Algebraic

  //                                ---------------  alignment Global CLHEP
  CLHEP::HepVector d3 = CLHEP::solve(Hess, -Grad);
  int iEr3;
  CLHEP::HepMatrix Errd3 = Hess.inverse(iEr3);

  // end of Global CLHEP

  //                                ----------------- alignment Local CLHEP
  CLHEP::HepVector dLI = CLHEP::solve(HessL, -GradL);
  int iErI;
  CLHEP::HepMatrix ErrdLI = HessL.inverse(iErI);

  // end of Local CLHEP

  // printout of final parameters
  std::cout << " ---- " << N_event << " event " << N_track << " tracks "
            << MuSelect << " ---- " << std::endl;
  std::cout << " ALCARECOMuAlCalIsolatedMu " << std::endl;
  std::cout << " Simulated shifts[cm] " << MuGlShift(1) << " " << MuGlShift(2) << " " << MuGlShift(3) << " "
            << " angles[rad] " << MuGlAngle(1) << " " << MuGlAngle(2) << " " << MuGlAngle(3) << " " << std::endl;
  std::cout << " d    " << -d << std::endl;
  ;
  std::cout << " +-   " << sqrt(InfI(0, 0)) << " " << sqrt(InfI(1, 1)) << " " << sqrt(InfI(2, 2)) << std::endl;
  std::cout << " dG   " << d3(1) << " " << d3(2) << " " << d3(3) << " " << d3(4) << " " << d3(5) << " " << d3(6)
            << std::endl;
  ;
  std::cout << " +-   " << sqrt(Errd3(1, 1)) << " " << sqrt(Errd3(2, 2)) << " " << sqrt(Errd3(3, 3)) << " "
            << sqrt(Errd3(4, 4)) << " " << sqrt(Errd3(5, 5)) << " " << sqrt(Errd3(6, 6)) << std::endl;
  std::cout << " dL   " << dLI(1) << " " << dLI(2) << " " << dLI(3) << " " << dLI(4) << " " << dLI(5) << " " << dLI(6)
            << std::endl;
  ;
  std::cout << " +-   " << sqrt(ErrdLI(1, 1)) << " " << sqrt(ErrdLI(2, 2)) << " " << sqrt(ErrdLI(3, 3)) << " "
            << sqrt(ErrdLI(4, 4)) << " " << sqrt(ErrdLI(5, 5)) << " " << sqrt(ErrdLI(6, 6)) << std::endl;

  // what do we write to DB
  CLHEP::HepVector vectorToDb(6, 0), vectorErrToDb(6, 0);
  vectorToDb = -dLI;
  for (unsigned int i = 1; i <= 6; i++)
    vectorErrToDb(i) = sqrt(ErrdLI(i, i));

  // write histograms to root file
  file->Write();
  file->Close();

  // write global parameters to text file
  OutGlobalTxt.open(txtOutFile_.c_str(), ios::out);
  if (!OutGlobalTxt.is_open())
    std::cout << " outglobal.txt is not open !!!!!" << std::endl;
  else {
    OutGlobalTxt << vectorToDb(1) << " " << vectorToDb(2) << " "
                 << vectorToDb(3) << " " << vectorToDb(4) << " "
                 << vectorToDb(5) << " " << vectorToDb(6) << " muon Global.\n";
    OutGlobalTxt << vectorErrToDb(1) << " " << vectorErrToDb(1) << " "
                 << vectorErrToDb(1) << " " << vectorErrToDb(1) << " "
                 << vectorErrToDb(1) << " " << vectorErrToDb(1) << " errors.\n";
    OutGlobalTxt << N_event << " events are processed.\n";
    OutGlobalTxt <<"HessL \n"<<HessL << "\n";
    OutGlobalTxt <<"GradL \n"<<GradL <<"\n";

    OutGlobalTxt << "ALCARECOMuAlCalIsolatedMu.\n";
    OutGlobalTxt.close();
    std::cout << " Write to the file outglobal.txt done  " << std::endl;
  }

  // write new GlobalPositionRcd to DB
  if (writeDB_)
    GlobalTrackerMuonAlignment::writeGlPosRcd(vectorToDb);
}

// ------------ book histogram  ------------
void GlobalTrackerMuonAlignment::bookHist() {
  double PI = 3.1415927;
  histo = new TH1F("Pt", "pt", 1000, 0, 100);
  histo2 = new TH1F("P", "P [GeV/c]", 400, 0., 400.);
  histo2->GetXaxis()->SetTitle("momentum [GeV/c]");
  histo3 = new TH1F("outerLambda", "#lambda outer", 100, -PI / 2., PI / 2.);
  histo3->GetXaxis()->SetTitle("#lambda outer");
  histo4 = new TH1F("phi", "#phi [rad]", 100, -PI, PI);
  histo4->GetXaxis()->SetTitle("#phi [rad]");
  histo5 = new TH1F("Rmuon", "inner muon hit R [cm]", 100, 0., 800.);
  histo5->GetXaxis()->SetTitle("R of muon [cm]");
  histo6 = new TH1F("Zmuon", "inner muon hit Z[cm]", 100, -1000., 1000.);
  histo6->GetXaxis()->SetTitle("Z of muon [cm]");
  histo7 = new TH1F("(Pm-Pt)/Pt", " (Pmuon-Ptrack)/Ptrack", 100, -2., 2.);
  histo7->GetXaxis()->SetTitle("(Pmuon-Ptrack)/Ptrack");
  histo8 = new TH1F("chi muon-track", "#chi^{2}(muon-track)", 1000, 0., 1000.);
  histo8->GetXaxis()->SetTitle("#chi^{2} of muon w.r.t. propagated track");
  histo11 = new TH1F("distance muon-track", "distance muon w.r.t track [cm]",
                     100, 0., 30.);
  histo11->GetXaxis()->SetTitle("distance of muon w.r.t. track [cm]");
  histo12 = new TH1F("Xmuon-Xtrack", "Xmuon-Xtrack [cm]", 200, -20., 20.);
  histo12->GetXaxis()->SetTitle("Xmuon - Xtrack [cm]");
  histo13 = new TH1F("Ymuon-Ytrack", "Ymuon-Ytrack [cm]", 200, -20., 20.);
  histo13->GetXaxis()->SetTitle("Ymuon - Ytrack [cm]");
  histo14 = new TH1F("Zmuon-Ztrack", "Zmuon-Ztrack [cm]", 200, -20., 20.);
  histo14->GetXaxis()->SetTitle("Zmuon-Ztrack [cm]");
  histo15 = new TH1F("NXmuon-NXtrack", "NXmuon-NXtrack [rad]", 200, -.1, .1);
  histo15->GetXaxis()->SetTitle("N_{X}(muon)-N_{X}(track) [rad]");
  histo16 = new TH1F("NYmuon-NYtrack", "NYmuon-NYtrack [rad]", 200, -.1, .1);
  histo16->GetXaxis()->SetTitle("N_{Y}(muon)-N_{Y}(track) [rad]");
  histo17 = new TH1F("NZmuon-NZtrack", "NZmuon-NZtrack [rad]", 200, -.1, .1);
  histo17->GetXaxis()->SetTitle("N_{Z}(muon)-N_{Z}(track) [rad]");
  histo18 = new TH1F("expected error of Xinner", "outer hit of inner tracker",
                     100, 0, .01);
  histo18->GetXaxis()->SetTitle("expected error of Xinner [cm]");
  histo19 =
      new TH1F("expected error of Xmuon", "inner hit of muon", 100, 0, .1);
  histo19->GetXaxis()->SetTitle("expected error of Xmuon [cm]");
  histo20 = new TH1F("expected error of Xmuon-Xtrack",
                     "muon w.r.t. propagated track", 100, 0., 10.);
  histo20->GetXaxis()->SetTitle("expected error of Xmuon-Xtrack [cm]");
  histo21 =
      new TH1F("pull of Xmuon-Xtrack", "pull of Xmuon-Xtrack", 100, -10., 10.);
  histo21->GetXaxis()->SetTitle("(Xmuon-Xtrack)/expected error");
  histo22 =
      new TH1F("pull of Ymuon-Ytrack", "pull of Ymuon-Ytrack", 100, -10., 10.);
  histo22->GetXaxis()->SetTitle("(Ymuon-Ytrack)/expected error");
  histo23 =
      new TH1F("pull of Zmuon-Ztrack", "pull of Zmuon-Ztrack", 100, -10., 10.);
  histo23->GetXaxis()->SetTitle("(Zmuon-Ztrack)/expected error");
  histo24 = new TH1F("pull of PXmuon-PXtrack", "pull of PXmuon-PXtrack", 100,
                     -10., 10.);
  histo24->GetXaxis()->SetTitle("(P_{X}(muon)-P_{X}(track))/expected error");
  histo25 = new TH1F("pull of PYmuon-PYtrack", "pull of PYmuon-PYtrack", 100,
                     -10., 10.);
  histo25->GetXaxis()->SetTitle("(P_{Y}(muon)-P_{Y}(track))/expected error");
  histo26 = new TH1F("pull of PZmuon-PZtrack", "pull of PZmuon-PZtrack", 100,
                     -10., 10.);
  histo26->GetXaxis()->SetTitle("(P_{Z}(muon)-P_{Z}(track))/expected error");
  histo27 = new TH1F("N_x", "Nx of tangent plane", 120, -1.1, 1.1);
  histo27->GetXaxis()->SetTitle("normal vector projection N_{X}");
  histo28 = new TH1F("N_y", "Ny of tangent plane", 120, -1.1, 1.1);
  histo28->GetXaxis()->SetTitle("normal vector projection N_{Y}");
  histo29 = new TH1F("lenght of track", "lenght of track", 200, 0., 400);
  histo29->GetXaxis()->SetTitle("lenght of track [cm]");
  histo30 = new TH1F("lenght of muon", "lenght of muon", 200, 0., 800);
  histo30->GetXaxis()->SetTitle("lenght of muon [cm]");

  histo31 = new TH1F("local chi muon-track", "#local chi^{2}(muon-track)", 1000,
                     0., 1000.);
  histo31->GetXaxis()->SetTitle(
      "#local chi^{2} of muon w.r.t. propagated track");
  histo32 =
      new TH1F("pull of Px/Pz local", "pull of Px/Pz local", 100, -10., 10.);
  histo32->GetXaxis()->SetTitle(
      "local (Px/Pz(muon) - Px/Pz(track))/expected error");
  histo33 =
      new TH1F("pull of Py/Pz local", "pull of Py/Pz local", 100, -10., 10.);
  histo33->GetXaxis()->SetTitle(
      "local (Py/Pz(muon) - Py/Pz(track))/expected error");
  histo34 = new TH1F("pull of X local", "pull of X local", 100, -10., 10.);
  histo34->GetXaxis()->SetTitle("local (Xmuon - Xtrack)/expected error");
  histo35 = new TH1F("pull of Y local", "pull of Y local", 100, -10., 10.);
  histo35->GetXaxis()->SetTitle("local (Ymuon - Ytrack)/expected error");

  histo101 = new TH2F("Rtr/mu vs Ztr/mu", "hit of track/muon", 100, -800., 800.,
                      100, 0., 600.);
  histo101->GetXaxis()->SetTitle("Z of track/muon [cm]");
  histo101->GetYaxis()->SetTitle("R of track/muon [cm]");
  histo102 = new TH2F("Ytr/mu vs Xtr/mu", "hit of track/muon", 100, -600., 600.,
                      100, -600., 600.);
  histo102->GetXaxis()->SetTitle("X of track/muon [cm]");
  histo102->GetYaxis()->SetTitle("Y of track/muon [cm]");
}

// ------------ fit histogram  ------------
void GlobalTrackerMuonAlignment::fitHist() {
  histo7->Fit("gaus", "Q");

  histo12->Fit("gaus", "Q");
  histo13->Fit("gaus", "Q");
  histo14->Fit("gaus", "Q");
  histo15->Fit("gaus", "Q");
  histo16->Fit("gaus", "Q");
  histo17->Fit("gaus", "Q");

  histo21->Fit("gaus", "Q");
  histo22->Fit("gaus", "Q");
  histo23->Fit("gaus", "Q");
  histo24->Fit("gaus", "Q");
  histo25->Fit("gaus", "Q");
  histo26->Fit("gaus", "Q");

  histo32->Fit("gaus", "Q");
  histo33->Fit("gaus", "Q");
  histo34->Fit("gaus", "Q");
  histo35->Fit("gaus", "Q");
}

// ------- method to analyze recoTrack & trajectoryMuon of Global Muon ------
void GlobalTrackerMuonAlignment::analyzeTrackTrajectory(
    const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  using namespace reco;

  double PI = 3.1415927;

  N_event++;
  edm::Handle<reco::MuonCollection> smuons;
  iEvent.getByToken(smuons_, smuons);
  edm::Handle<TrajTrackAssociationCollection> ref_track;
  iEvent.getByToken(ref_track_, ref_track);
  //   -----  Convert TrackPairs to reco::Track for refitted Tracker Track
    const Trajectory* trajVec;
    //const reco::Track* ref_track_vec;
    ConstTrajTrackPairs ref_track_pairs;
    for (auto iter = ref_track->begin(); iter != ref_track->end(); ++iter) {
      ref_track_pairs.push_back(
          ConstTrajTrackPair(&(*(*iter).key), &(*(*iter).val)));
    }
    for (ConstTrajTrackPairs::const_iterator trajtrack =
             ref_track_pairs.begin();
         trajtrack != ref_track_pairs.end(); ++trajtrack)

      {
      trajVec = (*trajtrack).first;
    //  ref_track_vec = (*trajtrack).second;
    }

  if (smuons->size() != 1)
    return;

  if (watchTrackingGeometry_.check(iSetup) || !trackingGeometry_) {
    edm::ESHandle<GlobalTrackingGeometry> trackingGeometry =
        iSetup.getHandle(m_TkGeometryToken);
    trackingGeometry_ = &*trackingGeometry;
    theTrackingGeometry = trackingGeometry;
  }

  if (watchMagneticFieldRecord_.check(iSetup) || !magneticField_) {
    edm::ESHandle<MagneticField> magneticField =
        iSetup.getHandle(m_MagFieldToken);
    magneticField_ = &*magneticField;
  }

  if (watchGlobalPositionRcd_.check(iSetup) || !globalPositionRcd_) {
    edm::ESHandle<Alignments> globalPositionRcd =
        iSetup.getHandle(m_globalPosToken);
    globalPositionRcd_ = &*globalPositionRcd;
    for (std::vector<AlignTransform>::const_iterator i =
             globalPositionRcd_->m_align.begin();
         i != globalPositionRcd_->m_align.end(); ++i) {
      if (DetId(DetId::Tracker).rawId() == i->rawId())
        iteratorTrackerRcd = i;
      if (DetId(DetId::Muon).rawId() == i->rawId())
        iteratorMuonRcd = i;
      if (DetId(DetId::Ecal).rawId() == i->rawId())
        iteratorEcalRcd = i;
      if (DetId(DetId::Hcal).rawId() == i->rawId())
        iteratorHcalRcd = i;
    }
  } // end of GlobalPositionRcd

  edm::ESHandle<Propagator> propagator = iSetup.getHandle(m_propToken);

  SteppingHelixPropagator alongStHePr =
      SteppingHelixPropagator(magneticField_, alongMomentum);
  SteppingHelixPropagator oppositeStHePr =
      SteppingHelixPropagator(magneticField_, oppositeToMomentum);

  defaultRKPropagator::Product aprod(magneticField_, alongMomentum, 5.e-5);
  auto &alongRKPr = aprod.propagator;
  defaultRKPropagator::Product oprod(magneticField_, oppositeToMomentum, 5.e-5);
  auto &oppositeRKPr = oprod.propagator;

  float epsilon = 5.;
  SmartPropagator alongSmPr = SmartPropagator(
      alongRKPr, alongStHePr, magneticField_, alongMomentum, epsilon);
  SmartPropagator oppositeSmPr =
      SmartPropagator(oppositeRKPr, oppositeStHePr, magneticField_,
                      oppositeToMomentum, epsilon);

  if (defineFitter) {
    KFUpdator *theUpdator = new KFUpdator();
    Chi2MeasurementEstimator *theEstimator =
        new Chi2MeasurementEstimator(100000, 100000);
    theFitter = new KFTrajectoryFitter(alongSmPr, *theUpdator, *theEstimator);
    theSmoother =
        new KFTrajectorySmoother(alongSmPr, *theUpdator, *theEstimator);
    theFitterOp =
        new KFTrajectoryFitter(oppositeSmPr, *theUpdator, *theEstimator);
    theSmootherOp =
        new KFTrajectorySmoother(oppositeSmPr, *theUpdator, *theEstimator);

    edm::ESHandle<TransientTrackingRecHitBuilder> builder =
        iSetup.getHandle(m_ttrhBuilderToken);

    this->TTRHBuilder = &(*builder);
    this->MuRHBuilder =
    new MuonTransientTrackingRecHitBuilder(theTrackingGeometry);
    std::cout<<" theTrackingGeometry.isValid() "<<theTrackingGeometry.isValid()<<std::endl;
    defineFitter = false; // was false
  }

  for (MuonCollection::const_iterator itMuon = smuons->begin();
       itMuon != smuons->end(); ++itMuon) {

    //If there will not be this skip then outermost or innermost state might be not valid
    //I'm not sure if "itMuon->isTrackerMuon()!= 1" should be here because I saw that some events with "itMuon->isTrackerMuon()!= 1" processed well
     if (itMuon->isGlobalMuon() != 1 ||itMuon->isMuon() != 1 || itMuon->isStandAloneMuon() != 1 || itMuon->isTrackerMuon()!= 1 )
     return;
    // get information about the innermost muon hit -------------------------
    TransientTrack muTT(itMuon->outerTrack(), magneticField_,
                        trackingGeometry_);
    TrajectoryStateOnSurface innerMuTSOS = muTT.innermostMeasurementState();
    TrajectoryStateOnSurface outerMuTSOS = muTT.outermostMeasurementState();
    // get information about the outermost tracker hit -----------------------
    TransientTrack trackTT(itMuon->track(), magneticField_, trackingGeometry_);
    TrajectoryStateOnSurface outerTrackTSOS =
        trackTT.outermostMeasurementState();
    TrajectoryStateOnSurface innerTrackTSOS =
        trackTT.innermostMeasurementState();
    GlobalPoint pointTrackIn = innerTrackTSOS.globalPosition();
    GlobalPoint pointTrackOut = outerTrackTSOS.globalPosition();
    float lenghtTrack = (pointTrackOut - pointTrackIn).mag();
    GlobalPoint pointMuonIn = innerMuTSOS.globalPosition();
    GlobalPoint pointMuonOut = outerMuTSOS.globalPosition();
    float lenghtMuon = (pointMuonOut - pointMuonIn).mag();
//    GlobalVector momentumTrackOut = outerTrackTSOS.globalMomentum();
//    GlobalVector momentumTrackIn = innerTrackTSOS.globalMomentum();
//debug
//     float distanceInIn = (pointTrackIn - pointMuonIn).mag();
//     float distanceInOut = (pointTrackIn - pointMuonOut).mag();
//     float distanceOutIn = (pointTrackOut - pointMuonIn).mag();
//     float distanceOutOut = (pointTrackOut - pointMuonOut).mag();
//
//       std::cout << " pointTrackIn " << pointTrackIn << std::endl;
//       std::cout << "          Out " << pointTrackOut << " lenght " << lenghtTrack << std::endl;
//       std::cout << " point MuonIn " << pointMuonIn << std::endl;
//       std::cout << "          Out " << pointMuonOut << " lenght " << lenghtMuon << std::endl;
//       std::cout << " momeTrackIn Out " << momentumTrackIn << " " << momentumTrackOut << std::endl;
//       std::cout << " doi io ii oo " << distanceOutIn << " " << distanceInOut << " " << distanceInIn << " "
//                 << distanceOutOut << std::endl;
// //debug


if (lenghtTrack < 90.)
  continue;
if (lenghtMuon < 300.)
  continue;
if (momentumTrackIn.mag() < 15. || momentumTrackOut.mag() < 15.) continue;
//if (momentumTrackIn.mag() < 5. || momentumTrackOut.mag() < 5.)
if (trackTT.charge() != muTT.charge()) continue;


    GlobalVector GRm, GPm, Nl, Rm, Pm, Rt, Pt, Rt0;
    AlgebraicSymMatrix66 Cm, C0, Ce, C1;

    TrajectoryStateOnSurface extrapolationT;
    int tsosMuonIf = 0;

    //------------------------------- Isolated Muon ---
    // Out-In --
    TrajectoryStateOnSurface trackFittedTSOS;
    bool IsolatedMuonCheck = true;
    if (IsolatedMuonCheck == true)
    {
    const Surface &refSurface = innerMuTSOS.surface();
    ConstReferenceCountingPointer<TangentPlane> tpMuLocal(
        refSurface.tangentPlane(innerMuTSOS.localPosition()));
    Nl = tpMuLocal->normalVector();
    ConstReferenceCountingPointer<TangentPlane> tpMuGlobal(
        refSurface.tangentPlane(innerMuTSOS.globalPosition()));
    //GlobalVector Ng = tpMuGlobal->normalVector();
    //Surface *surf = (Surface *)&refSurface;
    //const Plane *refPlane = dynamic_cast<Plane *>(surf);
    //GlobalVector Nlp = refPlane->normalVector();

    //                          extrapolation to innermost muon hit
    trackFittedTSOS = (*trajVec).lastMeasurement().updatedState();
    if (trackFittedTSOS.isValid())
    {
    extrapolationT = alongSmPr.propagate(trackFittedTSOS, refSurface);
    if (!extrapolationT.isValid()) continue;
    }
    else continue;


    //optional No Refit
    if (!refitTrack_)
    {
      extrapolationT = alongSmPr.propagate(outerTrackTSOS, refSurface);
      if (!extrapolationT.isValid()) continue;
    }

    tsosMuonIf = 1;
    Rt = GlobalVector((extrapolationT.globalPosition()).x(),
                      (extrapolationT.globalPosition()).y(),
                      (extrapolationT.globalPosition()).z());

    Pt = extrapolationT.globalMomentum();

    //                          global parameters of muon
    GRm = GlobalVector((innerMuTSOS.globalPosition()).x(),
                       (innerMuTSOS.globalPosition()).y(),
                       (innerMuTSOS.globalPosition()).z());
    GPm = innerMuTSOS.globalMomentum();

    Rt0 = GlobalVector((outerTrackTSOS.globalPosition()).x(),
                       (outerTrackTSOS.globalPosition()).y(),
                       (outerTrackTSOS.globalPosition()).z());
    Cm = AlgebraicSymMatrix66(extrapolationT.cartesianError().matrix() +
                              innerMuTSOS.cartesianError().matrix());
    C0 = AlgebraicSymMatrix66(outerTrackTSOS.cartesianError().matrix());
    Ce = AlgebraicSymMatrix66(extrapolationT.cartesianError().matrix());
    C1 = AlgebraicSymMatrix66(innerMuTSOS.cartesianError().matrix());
   }
    //                    -------------------------------   end Isolated Muon
    //                    -- Out - In  ---

    TrajectoryStateOnSurface tsosMuon;
    if (tsosMuonIf == 1)
      tsosMuon = muTT.innermostMeasurementState();
    else
      tsosMuon = muTT.outermostMeasurementState();

    AlgebraicVector4 LPRm; // muon local (dx/dz, dy/dz, x, y)
    GlobalTrackerMuonAlignment::misalignMuonL(GRm, GPm, Nl, Rt, Rm, Pm, LPRm,
                                              extrapolationT, tsosMuon);
    GlobalVector resR = Rm - Rt;
    GlobalVector resP0 = Pm - Pt;
    GlobalVector resP = Pm / Pm.mag() - Pt / Pt.mag();
    float RelMomResidual = (Pm.mag() - Pt.mag()) / (Pt.mag() + 1.e-6);

    AlgebraicVector6 Vm;
    Vm(0) = resR.x();
    Vm(1) = resR.y();
    Vm(2) = resR.z();
    Vm(3) = resP0.x();
    Vm(4) = resP0.y();
    Vm(5) = resP0.z();
    float Rmuon = Rm.perp();
    float Zmuon = Rm.z();
    double chi_d = 0;
    for (int i = 0; i <= 5; i++)
      chi_d += Vm(i) * Vm(i) / Cm(i, i);

    AlgebraicVector5 Vml(tsosMuon.localParameters().vector() -
                         extrapolationT.localParameters().vector());
    AlgebraicSymMatrix55 m(tsosMuon.localError().matrix() +
                           extrapolationT.localError().matrix());
    AlgebraicSymMatrix55 Cml(tsosMuon.localError().matrix() +
                             extrapolationT.localError().matrix());

    double chi_Loc = ROOT::Math::Similarity(Vml, m);

    if (Pt.mag() < 15.)
      continue;
    if (Pm.mag() < 5.)
      continue;

    if (fabs(resR.x()) > 20.)
      continue;
    if (fabs(resR.y()) > 20.)
      continue;
    if (fabs(resR.z()) > 20.)
      continue;
    if (fabs(resR.mag()) > 30.)
      continue;
    if (fabs(resP.x()) > 0.06)
      continue;
    if (fabs(resP.y()) > 0.06)
      continue;
    if (fabs(resP.z()) > 0.06)
      continue;
    if (chi_d > 40.)
      continue;

    //                                                 select Barrel, EndCap 1,
    //                                                 2
    if (Rmuon < 120. || Rmuon > 450.)
      continue;
    if (Zmuon < -720. || Zmuon > 720.)
      continue;
    MuSelect = " Barrel+EndCaps";

    N_track++;
    //                     gradient and Hessian for each track

    GlobalTrackerMuonAlignment::gradientGlobalAlg(Rt, Pt, Rm, Nl, Cm);
    GlobalTrackerMuonAlignment::gradientGlobal(Rt, Pt, Rm, Pm, Nl, Cm);

    CLHEP::HepSymMatrix covLoc(4, 0);
    for (int i = 1; i <= 4; i++)
      for (int j = 1; j <= i; j++) {
        covLoc(i, j) = (tsosMuon.localError().matrix() +
                        extrapolationT.localError().matrix())(i, j);
      }

    const Surface &refSurface = tsosMuon.surface();
    CLHEP::HepMatrix rotLoc(3, 3, 0);
    rotLoc(1, 1) = refSurface.rotation().xx();
    rotLoc(1, 2) = refSurface.rotation().xy();
    rotLoc(1, 3) = refSurface.rotation().xz();

    rotLoc(2, 1) = refSurface.rotation().yx();
    rotLoc(2, 2) = refSurface.rotation().yy();
    rotLoc(2, 3) = refSurface.rotation().yz();

    rotLoc(3, 1) = refSurface.rotation().zx();
    rotLoc(3, 2) = refSurface.rotation().zy();
    rotLoc(3, 3) = refSurface.rotation().zz();

    CLHEP::HepVector posLoc(3);
    posLoc(1) = refSurface.position().x();
    posLoc(2) = refSurface.position().y();
    posLoc(3) = refSurface.position().z();

    GlobalTrackerMuonAlignment::gradientLocal(Rt, Pt, Rm, Pm, Nl, covLoc,
                                              rotLoc, posLoc, LPRm);

    // -----------------------------------------------------  fill histogram
    histo->Fill(itMuon->track()->pt());
    histo2->Fill(Pt.mag());
    histo3->Fill((PI / 2. - itMuon->track()->outerTheta()));
    histo4->Fill(itMuon->track()->phi());
    histo5->Fill(Rmuon);
    histo6->Fill(Zmuon);
    histo7->Fill(RelMomResidual);
    histo8->Fill(chi_d);
    histo101->Fill(Zmuon, Rmuon);
    histo101->Fill(Rt0.z(), Rt0.perp());
    histo102->Fill(Rt0.x(), Rt0.y());
    histo102->Fill(Rm.x(), Rm.y());
    histo11->Fill(resR.mag());
    if (fabs(Nl.x()) < 0.98)
      histo12->Fill(resR.x());
    if (fabs(Nl.y()) < 0.98)
      histo13->Fill(resR.y());
    if (fabs(Nl.z()) < 0.98)
      histo14->Fill(resR.z());
    histo15->Fill(resP.x());
    histo16->Fill(resP.y());
    histo17->Fill(resP.z());
    if ((fabs(Nl.x()) < 0.98) && (fabs(Nl.y()) < 0.98) &&
        (fabs(Nl.z()) < 0.98)) {
      histo18->Fill(sqrt(C0(0, 0)));
      histo19->Fill(sqrt(C1(0, 0)));
      histo20->Fill(sqrt(C1(0, 0) + Ce(0, 0)));
    }
    if (fabs(Nl.x()) < 0.98)
      histo21->Fill(Vm(0) / sqrt(Cm(0, 0)));
    if (fabs(Nl.y()) < 0.98)
      histo22->Fill(Vm(1) / sqrt(Cm(1, 1)));
    if (fabs(Nl.z()) < 0.98)
      histo23->Fill(Vm(2) / sqrt(Cm(2, 2)));
    histo24->Fill(Vm(3) / sqrt(C1(3, 3) + Ce(3, 3)));
    histo25->Fill(Vm(4) / sqrt(C1(4, 4) + Ce(4, 4)));
    histo26->Fill(Vm(5) / sqrt(C1(5, 5) + Ce(5, 5)));
    histo27->Fill(Nl.x());
    histo28->Fill(Nl.y());
    histo29->Fill(lenghtTrack);
    histo30->Fill(lenghtMuon);
    histo31->Fill(chi_Loc);
    histo32->Fill(Vml(1) / sqrt(Cml(1, 1)));
    histo33->Fill(Vml(2) / sqrt(Cml(2, 2)));
    histo34->Fill(Vml(3) / sqrt(Cml(3, 3)));
    histo35->Fill(Vml(4) / sqrt(Cml(4, 4)));
    AlgebraicSymMatrix66 Ro;
    double Diag[6];
    for (int i = 0; i <= 5; i++)
      Diag[i] = sqrt(Cm(i, i));
    for (int i = 0; i <= 5; i++)
      for (int j = 0; j <= 5; j++)
        Ro(i, j) = Cm(i, j) / Diag[i] / Diag[j];
    std::cout << " correlation matrix " << std::endl;
    std::cout << Ro << std::endl;

    AlgebraicSymMatrix66 CmI;
    for (int i = 0; i <= 5; i++)
      for (int j = 0; j <= 5; j++)
        CmI(i, j) = Cm(i, j);

    bool ierr = !CmI.Invert();
    if (ierr) {
      continue;
    }
    std::cout << " inverse cov matrix " << std::endl;
    std::cout << Cm << std::endl;

    double chi = ROOT::Math::Similarity(Vm, CmI);
    std::cout << " chi chi_d " << chi << " " << chi_d << std::endl;


} // end loop on selected muons, i.e. Jim's globalMuon

} // end of analyzeTrackTrajectory

// ----  calculate gradient and Hessian matrix (algebraic) to search global
// shifts ------
void GlobalTrackerMuonAlignment::gradientGlobalAlg(GlobalVector &Rt,
                                                   GlobalVector &Pt,
                                                   GlobalVector &Rm,
                                                   GlobalVector &Nl,
                                                   AlgebraicSymMatrix66 &Cm) {
  AlgebraicMatrix33 Jac;
  AlgebraicVector3 Wi, R_m, R_t, P_t, Norm, dR;

  R_m(0) = Rm.x();
  R_m(1) = Rm.y();
  R_m(2) = Rm.z();
  R_t(0) = Rt.x();
  R_t(1) = Rt.y();
  R_t(2) = Rt.z();
  P_t(0) = Pt.x();
  P_t(1) = Pt.y();
  P_t(2) = Pt.z();
  Norm(0) = Nl.x();
  Norm(1) = Nl.y();
  Norm(2) = Nl.z();

  for (int i = 0; i <= 2; i++) {
    if (Cm(i, i) > 1.e-20)
      Wi(i) = 1. / Cm(i, i);
    else
      Wi(i) = 1.e-10;
    dR(i) = R_m(i) - R_t(i);
  }

  float PtN = P_t(0) * Norm(0) + P_t(1) * Norm(1) + P_t(2) * Norm(2);

  Jac(0, 0) = 1. - P_t(0) * Norm(0) / PtN;
  Jac(0, 1) = -P_t(0) * Norm(1) / PtN;
  Jac(0, 2) = -P_t(0) * Norm(2) / PtN;

  Jac(1, 0) = -P_t(1) * Norm(0) / PtN;
  Jac(1, 1) = 1. - P_t(1) * Norm(1) / PtN;
  Jac(1, 2) = -P_t(1) * Norm(2) / PtN;

  Jac(2, 0) = -P_t(2) * Norm(0) / PtN;
  Jac(2, 1) = -P_t(2) * Norm(1) / PtN;
  Jac(2, 2) = 1. - P_t(2) * Norm(2) / PtN;

  AlgebraicSymMatrix33 Itr;

  for (int i = 0; i <= 2; i++)
    for (int j = 0; j <= 2; j++) {
      if (j < i)
        continue;
      Itr(i, j) = 0.;
      for (int k = 0; k <= 2; k++) {
        Itr(i, j) += Jac(k, i) * Wi(k) * Jac(k, j);
      }
    }

  for (int i = 0; i <= 2; i++)
    for (int j = 0; j <= 2; j++) {
      if (j < i)
        continue;
      Inf(i, j) += Itr(i, j);
    }

  AlgebraicVector3 Gtr(0., 0., 0.);
  for (int i = 0; i <= 2; i++)
    for (int k = 0; k <= 2; k++)
      Gtr(i) += dR(k) * Wi(k) * Jac(k, i);
  for (int i = 0; i <= 2; i++)
    Gfr(i) += Gtr(i);


  return;
}

// ----  calculate gradient and Hessian matrix in global parameters ------
void GlobalTrackerMuonAlignment::gradientGlobal(
    GlobalVector &GRt, GlobalVector &GPt, GlobalVector &GRm, GlobalVector &GPm,
    GlobalVector &GNorm, AlgebraicSymMatrix66 &GCov) {

  int Nd = 6; // dimension of vector of alignment pararmeters, d

  CLHEP::HepSymMatrix w(Nd, 0);
  for (int i = 1; i <= Nd; i++)
    for (int j = 1; j <= Nd; j++) {
      if (j <= i)
        w(i, j) = GCov(i - 1, j - 1);
      if ((i == j) && (i <= 3) && (GCov(i - 1, j - 1) < 1.e-20))
        w(i, j) = 1.e20; // w=0
      if (i != j)
        w(i, j) = 0.; // use diaginal elements
    }

  CLHEP::HepVector V(Nd), Rt(3), Pt(3), Rm(3), Pm(3), Norm(3);
  Rt(1) = GRt.x();
  Rt(2) = GRt.y();
  Rt(3) = GRt.z();
  Pt(1) = GPt.x();
  Pt(2) = GPt.y();
  Pt(3) = GPt.z();
  Rm(1) = GRm.x();
  Rm(2) = GRm.y();
  Rm(3) = GRm.z();
  Pm(1) = GPm.x();
  Pm(2) = GPm.y();
  Pm(3) = GPm.z();
  Norm(1) = GNorm.x();
  Norm(2) = GNorm.y();
  Norm(3) = GNorm.z();

  V = dsum(Rm - Rt, Pm - Pt);

  double PmN = CLHEP_dot(Pm, Norm);

  CLHEP::HepMatrix Jac(Nd, Nd, 0);
  for (int i = 1; i <= 3; i++)
    for (int j = 1; j <= 3; j++) {
      Jac(i, j) = Pm(i) * Norm(j) / PmN;
      if (i == j)
        Jac(i, j) -= 1.;
    }

  //                                            dp/da
  Jac(4, 4) = 0.;     // dpx/dax
  Jac(5, 4) = -Pm(3); // dpy/dax
  Jac(6, 4) = Pm(2);  // dpz/dax
  Jac(4, 5) = Pm(3);  // dpx/day
  Jac(5, 5) = 0.;     // dpy/day
  Jac(6, 5) = -Pm(1); // dpz/day
  Jac(4, 6) = -Pm(2); // dpx/daz
  Jac(5, 6) = Pm(1);  // dpy/daz
  Jac(6, 6) = 0.;     // dpz/daz

  CLHEP::HepVector dsda(3);
  dsda(1) = (Norm(2) * Rm(3) - Norm(3) * Rm(2)) / PmN;
  dsda(2) = (Norm(3) * Rm(1) - Norm(1) * Rm(3)) / PmN;
  dsda(3) = (Norm(1) * Rm(2) - Norm(2) * Rm(1)) / PmN;

  //                                             dr/da
  Jac(1, 4) = Pm(1) * dsda(1);          // drx/dax
  Jac(2, 4) = -Rm(3) + Pm(2) * dsda(1); // dry/dax
  Jac(3, 4) = Rm(2) + Pm(3) * dsda(1);  // drz/dax

  Jac(1, 5) = Rm(3) + Pm(1) * dsda(2);  // drx/day
  Jac(2, 5) = Pm(2) * dsda(2);          // dry/day
  Jac(3, 5) = -Rm(1) + Pm(3) * dsda(2); // drz/day

  Jac(1, 6) = -Rm(2) + Pm(1) * dsda(3); // drx/daz
  Jac(2, 6) = Rm(1) + Pm(2) * dsda(3);  // dry/daz
  Jac(3, 6) = Pm(3) * dsda(3);          // drz/daz

  CLHEP::HepSymMatrix W(Nd, 0);
  int ierr;
  W = w.inverse(ierr);


  CLHEP::HepMatrix W_Jac(Nd, Nd, 0);
  W_Jac = Jac.T() * W;

  CLHEP::HepVector grad3(Nd);
  grad3 = W_Jac * V;

  CLHEP::HepMatrix hess3(Nd, Nd);
  hess3 = Jac.T() * W * Jac;

  Grad += grad3;
  Hess += hess3;

  CLHEP::HepVector d3I = CLHEP::solve(Hess, -Grad);
  int iEr3I;
  CLHEP::HepMatrix Errd3I = Hess.inverse(iEr3I);

  return;
} // end gradientGlobal

// ----  calculate gradient and Hessian matrix in local parameters ------
void GlobalTrackerMuonAlignment::gradientLocal(
    GlobalVector &GRt, GlobalVector &GPt, GlobalVector &GRm, GlobalVector &GPm,
    GlobalVector &GNorm, CLHEP::HepSymMatrix &covLoc, CLHEP::HepMatrix &rotLoc,
    CLHEP::HepVector &R0, AlgebraicVector4 &LPRm) {
  // we search for 6D global correction vector (d, a), where
  //                        3D vector of shihts d
  //               3D vector of rotation angles    a




  CLHEP::HepVector Rt(3), Pt(3), Rm(3), Pm(3), Norm(3);
  Rt(1) = GRt.x();
  Rt(2) = GRt.y();
  Rt(3) = GRt.z();
  Pt(1) = GPt.x();
  Pt(2) = GPt.y();
  Pt(3) = GPt.z();
  Rm(1) = GRm.x();
  Rm(2) = GRm.y();
  Rm(3) = GRm.z();
  Pm(1) = GPm.x();
  Pm(2) = GPm.y();
  Pm(3) = GPm.z();
  Norm(1) = GNorm.x();
  Norm(2) = GNorm.y();
  Norm(3) = GNorm.z();

  CLHEP::HepVector V(4), Rml(3), Pml(3), Rtl(3), Ptl(3);

  Rml = rotLoc * (Rm - R0);
  Rtl = rotLoc * (Rt - R0);
  Pml = rotLoc * Pm;
  Ptl = rotLoc * Pt;

  V(1) = LPRm(0) - Ptl(1) / Ptl(3);
  V(2) = LPRm(1) - Ptl(2) / Ptl(3);
  V(3) = LPRm(2) - Rtl(1);
  V(4) = LPRm(3) - Rtl(2);

  CLHEP::HepSymMatrix W = covLoc;

  int ierr;
  W.invert(ierr);

  //                               JacobianCartesianToLocal

  CLHEP::HepMatrix JacToLoc(4, 6, 0);
  for (int i = 1; i <= 2; i++)
    for (int j = 1; j <= 3; j++) {
      JacToLoc(i, j + 3) =
          (rotLoc(i, j) - rotLoc(3, j) * Pml(i) / Pml(3)) / Pml(3);
      JacToLoc(i + 2, j) = rotLoc(i, j);
    }

  //                                    JacobianCorrectionsToCartesian
  double PmN = CLHEP_dot(Pm, Norm);

  CLHEP::HepMatrix Jac(6, 6, 0);
  for (int i = 1; i <= 3; i++)
    for (int j = 1; j <= 3; j++) {
      Jac(i, j) = Pm(i) * Norm(j) / PmN;
      if (i == j)
        Jac(i, j) -= 1.;
    }

  //                                            dp/da
  Jac(4, 4) = 0.;     // dpx/dax
  Jac(5, 4) = -Pm(3); // dpy/dax
  Jac(6, 4) = Pm(2);  // dpz/dax
  Jac(4, 5) = Pm(3);  // dpx/day
  Jac(5, 5) = 0.;     // dpy/day
  Jac(6, 5) = -Pm(1); // dpz/day
  Jac(4, 6) = -Pm(2); // dpx/daz
  Jac(5, 6) = Pm(1);  // dpy/daz
  Jac(6, 6) = 0.;     // dpz/daz

  CLHEP::HepVector dsda(3);
  dsda(1) = (Norm(2) * Rm(3) - Norm(3) * Rm(2)) / PmN;
  dsda(2) = (Norm(3) * Rm(1) - Norm(1) * Rm(3)) / PmN;
  dsda(3) = (Norm(1) * Rm(2) - Norm(2) * Rm(1)) / PmN;

  //                                             dr/da
  Jac(1, 4) = Pm(1) * dsda(1);          // drx/dax
  Jac(2, 4) = -Rm(3) + Pm(2) * dsda(1); // dry/dax
  Jac(3, 4) = Rm(2) + Pm(3) * dsda(1);  // drz/dax

  Jac(1, 5) = Rm(3) + Pm(1) * dsda(2);  // drx/day
  Jac(2, 5) = Pm(2) * dsda(2);          // dry/day
  Jac(3, 5) = -Rm(1) + Pm(3) * dsda(2); // drz/day

  Jac(1, 6) = -Rm(2) + Pm(1) * dsda(3); // drx/daz
  Jac(2, 6) = Rm(1) + Pm(2) * dsda(3);  // dry/daz
  Jac(3, 6) = Pm(3) * dsda(3);          // drz/daz

  //                                   JacobianCorrectionToLocal
  CLHEP::HepMatrix JacCorLoc(4, 6, 0);
  JacCorLoc = JacToLoc * Jac;

  //                                   gradient and Hessian
  CLHEP::HepMatrix W_Jac(6, 4, 0);
  W_Jac = JacCorLoc.T() * W;

  CLHEP::HepVector gradL(6);
  gradL = W_Jac * V;

  CLHEP::HepMatrix hessL(6, 6);
  hessL = JacCorLoc.T() * W * JacCorLoc;

  GradL += gradL;
  HessL += hessL;

  CLHEP::HepVector dLI = CLHEP::solve(HessL, -GradL);
  int iErI;
  CLHEP::HepMatrix ErrdLI = HessL.inverse(iErI);

  return;
} // end gradientLocal

// ----  simulate a misalignment of muon system   ------
void GlobalTrackerMuonAlignment::misalignMuon(
    GlobalVector &GRm, GlobalVector &GPm, GlobalVector &Nl, GlobalVector &Rt,
    GlobalVector &Rm, GlobalVector &Pm) {
  CLHEP::HepVector d(3, 0), a(3, 0), Norm(3), Rm0(3), Pm0(3), Rm1(3), Pm1(3),
      Rmi(3), Pmi(3);

  d(1) = 0.0;
  d(2) = 0.0;
  d(3) = 0.0;
  a(1) = 0.0000;
  a(2) = 0.0000;
  a(3) = 0.0000; // zero

  MuGlShift = d;
  MuGlAngle = a;

  if ((d(1) == 0.) & (d(2) == 0.) & (d(3) == 0.) & (a(1) == 0.) & (a(2) == 0.) &
      (a(3) == 0.)) {
    Rm = GRm;
    Pm = GPm;
    return;
  }

  Rm0(1) = GRm.x();
  Rm0(2) = GRm.y();
  Rm0(3) = GRm.z();
  Pm0(1) = GPm.x();
  Pm0(2) = GPm.y();
  Pm0(3) = GPm.z();
  Norm(1) = Nl.x();
  Norm(2) = Nl.y();
  Norm(3) = Nl.z();

  CLHEP::HepMatrix T(3, 3, 1);

  double s1 = std::sin(a(1)), c1 = std::cos(a(1));
  double s2 = std::sin(a(2)), c2 = std::cos(a(2));
  double s3 = std::sin(a(3)), c3 = std::cos(a(3));
  T(1, 1) = c2 * c3;
  T(1, 2) = c1 * s3 + s1 * s2 * c3;
  T(1, 3) = s1 * s3 - c1 * s2 * c3;
  T(2, 1) = -c2 * s3;
  T(2, 2) = c1 * c3 - s1 * s2 * s3;
  T(2, 3) = s1 * c3 + c1 * s2 * s3;
  T(3, 1) = s2;
  T(3, 2) = -s1 * c2;
  T(3, 3) = c1 * c2;

  CLHEP::HepMatrix Ti = T.T();
  CLHEP::HepVector di = -Ti * d;

  CLHEP::HepVector ex0(3, 0), ey0(3, 0), ez0(3, 0), ex(3, 0), ey(3, 0),
      ez(3, 0);
  ex0(1) = 1.;
  ey0(2) = 1.;
  ez0(3) = 1.;
  ex = Ti * ex0;
  ey = Ti * ey0;
  ez = Ti * ez0;

  CLHEP::HepVector TiN = Ti * Norm;

  double si = CLHEP_dot((Ti * Rm0 - Rm0 + di), TiN) / CLHEP_dot(Pm0, TiN);
  Rm1(1) = CLHEP_dot(ex, Rm0 + si * Pm0 - di);
  Rm1(2) = CLHEP_dot(ey, Rm0 + si * Pm0 - di);
  Rm1(3) = CLHEP_dot(ez, Rm0 + si * Pm0 - di);
  Pm1 = T * Pm0;

  Rm = GlobalVector(Rm1(1), Rm1(2), Rm1(3));
  Pm = GlobalVector(CLHEP_dot(Pm0, ex), CLHEP_dot(Pm0, ey),
                    CLHEP_dot(Pm0, ez)); // =T*Pm0

  return;

} // end of misalignMuon

// ----  simulate a misalignment of muon system with Local  ------
void GlobalTrackerMuonAlignment::misalignMuonL(
    GlobalVector &GRm, GlobalVector &GPm, GlobalVector &Nl, GlobalVector &Rt,
    GlobalVector &Rm, GlobalVector &Pm, AlgebraicVector4 &Vm,
    TrajectoryStateOnSurface &tsosTrack, TrajectoryStateOnSurface &tsosMuon) {
  CLHEP::HepVector d(3, 0), a(3, 0), Norm(3), Rm0(3), Pm0(3), Rm1(3), Pm1(3),
      Rmi(3), Pmi(3);

  d(1) = 0.0;
  d(2) = 0.0;
  d(3) = 0.0;
  a(1) = 0.0000;
  a(2) = 0.0000;
  a(3) = 0.0000; // zero

  MuGlShift = d;
  MuGlAngle = a;

  if ((d(1) == 0.) & (d(2) == 0.) & (d(3) == 0.) & (a(1) == 0.) & (a(2) == 0.) &
      (a(3) == 0.)) {
    Rm = GRm;
    Pm = GPm;
    AlgebraicVector4 Vm0;
    Vm = AlgebraicVector4(tsosMuon.localParameters().vector()(1),
                          tsosMuon.localParameters().vector()(2),
                          tsosMuon.localParameters().vector()(3),
                          tsosMuon.localParameters().vector()(4));

    return;
  }

  Rm0(1) = GRm.x();
  Rm0(2) = GRm.y();
  Rm0(3) = GRm.z();
  Pm0(1) = GPm.x();
  Pm0(2) = GPm.y();
  Pm0(3) = GPm.z();
  Norm(1) = Nl.x();
  Norm(2) = Nl.y();
  Norm(3) = Nl.z();

  CLHEP::HepMatrix T(3, 3, 1);

  double s1 = std::sin(a(1)), c1 = std::cos(a(1));
  double s2 = std::sin(a(2)), c2 = std::cos(a(2));
  double s3 = std::sin(a(3)), c3 = std::cos(a(3));
  T(1, 1) = c2 * c3;
  T(1, 2) = c1 * s3 + s1 * s2 * c3;
  T(1, 3) = s1 * s3 - c1 * s2 * c3;
  T(2, 1) = -c2 * s3;
  T(2, 2) = c1 * c3 - s1 * s2 * s3;
  T(2, 3) = s1 * c3 + c1 * s2 * s3;
  T(3, 1) = s2;
  T(3, 2) = -s1 * c2;
  T(3, 3) = c1 * c2;

  //                    Ti, di what we have to apply for misalignment

  CLHEP::HepMatrix Ti = T.T();
  CLHEP::HepVector di = -Ti * d;
  CLHEP::HepVector ex0(3, 0), ey0(3, 0), ez0(3, 0), ex(3, 0), ey(3, 0),
      ez(3, 0);
  ex0(1) = 1.;
  ey0(2) = 1.;
  ez0(3) = 1.;
  ex = Ti * ex0;
  ey = Ti * ey0;
  ez = Ti * ez0;

  CLHEP::HepVector TiN = Ti * Norm;

  double si = CLHEP_dot((Ti * Rm0 - Rm0 + di), TiN) / CLHEP_dot(Pm0, TiN);
  Rm1(1) = CLHEP_dot(ex, Rm0 + si * Pm0 - di);
  Rm1(2) = CLHEP_dot(ey, Rm0 + si * Pm0 - di);
  Rm1(3) = CLHEP_dot(ez, Rm0 + si * Pm0 - di);
  Pm1 = T * Pm0;

  // Global muon parameters after misalignment
  Rm = GlobalVector(Rm1(1), Rm1(2), Rm1(3));
  Pm = GlobalVector(CLHEP_dot(Pm0, ex), CLHEP_dot(Pm0, ey),
                    CLHEP_dot(Pm0, ez)); // =T*Pm0

  // Local muon parameters after misalignment
  const Surface &refSurface = tsosMuon.surface();
  CLHEP::HepMatrix Tl(3, 3, 0);

  Tl(1, 1) = refSurface.rotation().xx();
  Tl(1, 2) = refSurface.rotation().xy();
  Tl(1, 3) = refSurface.rotation().xz();

  Tl(2, 1) = refSurface.rotation().yx();
  Tl(2, 2) = refSurface.rotation().yy();
  Tl(2, 3) = refSurface.rotation().yz();

  Tl(3, 1) = refSurface.rotation().zx();
  Tl(3, 2) = refSurface.rotation().zy();
  Tl(3, 3) = refSurface.rotation().zz();

  CLHEP::HepVector R0(3, 0), newR0(3, 0), newRl(3, 0), newPl(3, 0);
  R0(1) = refSurface.position().x();
  R0(2) = refSurface.position().y();
  R0(3) = refSurface.position().z();

  newRl = Tl * (Rm1 - R0);
  newPl = Tl * Pm1;

  Vm(0) = newPl(1) / newPl(3);
  Vm(1) = newPl(2) / newPl(3);
  Vm(2) = newRl(1);
  Vm(3) = newRl(2);

  return;

} // end misalignMuonL

// ----  write GlobalPositionRcd   ------
void GlobalTrackerMuonAlignment::writeGlPosRcd(CLHEP::HepVector &paramVec) {

  std::cout << " paramVector " << paramVec.T() << std::endl;
  CLHEP::Hep3Vector colX, colY, colZ;


    double s1 = std::sin(paramVec(4)), c1 = std::cos(paramVec(4));
    double s2 = std::sin(paramVec(5)), c2 = std::cos(paramVec(5));
    double s3 = std::sin(paramVec(6)), c3 = std::cos(paramVec(6));
    colX = CLHEP::Hep3Vector(               c2 * c3,               -c2 * s3,       s2);
    colY = CLHEP::Hep3Vector(c1 * s3 + s1 * s2 * c3, c1 * c3 - s1 * s2 * s3, -s1 * c2);
    colZ = CLHEP::Hep3Vector(s1 * s3 - c1 * s2 * c3, s1 * c3 + c1 * s2 * s3,  c1 * c2);


  CLHEP::HepVector param0(6, 0);

  Alignments globalPositions{};

  // Tracker
  AlignTransform tracker(iteratorTrackerRcd->translation(),
                         iteratorTrackerRcd->rotation(),
                         DetId(DetId::Tracker).rawId());
  // Muon
  CLHEP::Hep3Vector posMuGlRcd = iteratorMuonRcd->translation();
  CLHEP::HepRotation rotMuGlRcd = iteratorMuonRcd->rotation();
  CLHEP::HepEulerAngles angMuGlRcd = iteratorMuonRcd->rotation().eulerAngles();

  AlignTransform muon;
  if ((angMuGlRcd.phi() == 0.) && (angMuGlRcd.theta() == 0.) &&
      (angMuGlRcd.psi() == 0.) && (posMuGlRcd.x() == 0.) &&
      (posMuGlRcd.y() == 0.) && (posMuGlRcd.z() == 0.)) {
    std::cout << " New muon parameters are stored in Rcd" << std::endl;

    AlignTransform muonNew(
        AlignTransform::Translation(paramVec(1), paramVec(2), paramVec(3)),
        AlignTransform::Rotation(colX, colY, colZ), DetId(DetId::Muon).rawId());
    muon = muonNew;
  } else if ((paramVec(1) == 0.) && (paramVec(2) == 0.) &&
             (paramVec(3) == 0.) && (paramVec(4) == 0.) &&
             (paramVec(5) == 0.) && (paramVec(6) == 0.)) {
    std::cout << " Old muon parameters are stored in Rcd" << std::endl;

    AlignTransform muonNew(iteratorMuonRcd->translation(),
                           iteratorMuonRcd->rotation(),
                           DetId(DetId::Muon).rawId());
    muon = muonNew;
  } else {
    std::cout << " New + Old muon parameters are stored in Rcd" << std::endl;
    CLHEP::Hep3Vector posMuGlRcdThis =
        CLHEP::Hep3Vector(paramVec(1), paramVec(2), paramVec(3));
    CLHEP::HepRotation rotMuGlRcdThis = CLHEP::HepRotation(colX, colY, colZ);
    CLHEP::Hep3Vector posMuGlRcdNew =
        rotMuGlRcdThis * posMuGlRcd + posMuGlRcdThis;
    CLHEP::HepRotation rotMuGlRcdNew = rotMuGlRcdThis * rotMuGlRcd;

    AlignTransform muonNew(posMuGlRcdNew, rotMuGlRcdNew,
                           DetId(DetId::Muon).rawId());
    muon = muonNew;
  }

  // Ecal
  AlignTransform ecal(iteratorEcalRcd->translation(),
                      iteratorEcalRcd->rotation(), DetId(DetId::Ecal).rawId());
  // Hcal
  AlignTransform hcal(iteratorHcalRcd->translation(),
                      iteratorHcalRcd->rotation(), DetId(DetId::Hcal).rawId());
  // Calo
  AlignTransform calo(
      AlignTransform::Translation(param0(1), param0(2), param0(3)),
      AlignTransform::EulerAngles(param0(4), param0(5), param0(6)),
      DetId(DetId::Calo).rawId());

  std::cout << "Tracker (" << tracker.rawId() << ") at "
            << tracker.translation() << " " << tracker.rotation().eulerAngles()
            << std::endl;
  std::cout << tracker.rotation() << std::endl;

  std::cout << "Muon (" << muon.rawId() << ") at " << muon.translation() << " "
            << muon.rotation().eulerAngles() << std::endl;
  std::cout << "          rotations angles around x,y,z "
            << " ( " << -muon.rotation().zy() << " " << muon.rotation().zx()
            << " " << -muon.rotation().yx() << " )" << std::endl;
  std::cout << muon.rotation() << std::endl;

  std::cout << "Ecal (" << ecal.rawId() << ") at " << ecal.translation() << " "
            << ecal.rotation().eulerAngles() << std::endl;
  std::cout << ecal.rotation() << std::endl;

  std::cout << "Hcal (" << hcal.rawId() << ") at " << hcal.translation() << " "
            << hcal.rotation().eulerAngles() << std::endl;
  std::cout << hcal.rotation() << std::endl;

  std::cout << "Calo (" << calo.rawId() << ") at " << calo.translation() << " "
            << calo.rotation().eulerAngles() << std::endl;
  std::cout << calo.rotation() << std::endl;

  globalPositions.m_align.push_back(tracker);
  globalPositions.m_align.push_back(muon);
  globalPositions.m_align.push_back(ecal);
  globalPositions.m_align.push_back(hcal);
  globalPositions.m_align.push_back(calo);

  std::cout << "Uploading to the database..." << std::endl;
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
  poolDbService->writeOneIOV<Alignments>(
      globalPositions, poolDbService->currentTime(), "GlobalPositionRcd");
  std::cout << "done!" << std::endl;

  return;
}

// define this as a plug-in
DEFINE_FWK_MODULE(GlobalTrackerMuonAlignment);
