// -*- C++ -*-
//
// Package:    GlobalTrackerMuonAlignment
// Class:      GlobalTrackerMuonAlignment
// 
/**\class GlobalTrackerMuonAlignment GlobalTrackerMuonAlignment.cc Alignment/GlobalTrackerMuonAlignment/src/GlobalTrackerMuonAlignment.cc

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

// system include files
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <typeinfo>

// user include files

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

// references
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/SmartPropagator.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirectionChooser.h" 
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"

// Database
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"

// Refit
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include <TH1.h>
#include <TH2.h>
#include <TFile.h>

#include<iostream>
#include<cmath>
#include<assert.h>
#include<fstream>
#include<iomanip>

using namespace edm; 
using namespace std;
using namespace reco;

//
// class declaration
//

class GlobalTrackerMuonAlignment : public edm::EDAnalyzer {
 public:
  explicit GlobalTrackerMuonAlignment(const edm::ParameterSet&);
  ~GlobalTrackerMuonAlignment();

  void analyzeTrackTrack(const edm::Event&, const edm::EventSetup&);
  void analyzeTrackTrajectory(const edm::Event&, const edm::EventSetup&);
  void bookHist();
  void fitHist();
  void trackFitter(reco::TrackRef, reco::TransientTrack&, PropagationDirection, 
		  TrajectoryStateOnSurface&);
  void muonFitter(reco::TrackRef, reco::TransientTrack&, PropagationDirection, 
		  TrajectoryStateOnSurface&);
  void debugTrackHit(const std::string, reco::TrackRef);
  void debugTrackHit(const std::string, reco::TransientTrack&);
  void debugTrajectorySOS(const std::string, TrajectoryStateOnSurface&);
  void debugTrajectorySOSv(const std::string, TrajectoryStateOnSurface);
  void debugTrajectory(const std::string, Trajectory&);

  void gradientGlobal(GlobalVector&, GlobalVector&, GlobalVector&, GlobalVector&, 
		      GlobalVector&, AlgebraicSymMatrix66&);
  void gradientLocal(GlobalVector&, GlobalVector&, GlobalVector&, GlobalVector&, 
		     GlobalVector&, CLHEP::HepSymMatrix&, CLHEP::HepMatrix&, 
		     CLHEP::HepVector&, AlgebraicVector4&);
  void gradientGlobalAlg(GlobalVector&, GlobalVector&, GlobalVector&, GlobalVector&,  
  			 AlgebraicSymMatrix66&);
  void misalignMuon(GlobalVector&, GlobalVector&, GlobalVector&, GlobalVector&, 
		    GlobalVector&, GlobalVector&);
  void misalignMuonL(GlobalVector&, GlobalVector&, GlobalVector&, GlobalVector&, 
		     GlobalVector&, GlobalVector&, AlgebraicVector4&, 
		     TrajectoryStateOnSurface&, TrajectoryStateOnSurface&);
  void writeGlPosRcd(CLHEP::HepVector& d3); 
  inline double CLHEP_dot(const CLHEP::HepVector& a, const CLHEP::HepVector& b)
  {
    return a(1)*b(1)+a(2)*b(2)+a(3)*b(3);
  }

 private:

  virtual void beginJob() override ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;
  
  // ----------member data ---------------------------

  edm::InputTag trackTags_;     // used to select what tracks to read from configuration file
  edm::InputTag muonTags_;      // used to select what standalone muons
  edm::InputTag gmuonTags_;     // used to select what global muons
  edm::InputTag smuonTags_;     // used to select what selected muons
  string propagator_;           // name of the propagator
  bool cosmicMuonMode_;   
  bool isolatedMuonMode_;
  bool collectionCosmic;        // if true, read muon collection expected for CosmicMu
  bool collectionIsolated;      //                                            IsolateMu
  bool refitMuon_;              // if true, refit stand alone muon track
  bool refitTrack_;             //                tracker track
  string rootOutFile_;
  string txtOutFile_;
  bool writeDB_;                // write results to DB
  bool debug_;                  // run in debugging mode

  edm::ESWatcher<GlobalTrackingGeometryRecord> watchTrackingGeometry_;
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  const GlobalTrackingGeometry* trackingGeometry_;

  edm::ESWatcher<IdealMagneticFieldRecord> watchMagneticFieldRecord_;
  const MagneticField* magneticField_;

  edm::ESWatcher<TrackingComponentsRecord> watchTrackingComponentsRecord_;

  edm::ESWatcher<GlobalPositionRcd> watchGlobalPositionRcd_;
  const Alignments* globalPositionRcd_;

  KFTrajectoryFitter* theFitter;
  KFTrajectorySmoother* theSmoother;
  KFTrajectoryFitter* theFitterOp;
  KFTrajectorySmoother* theSmootherOp;
  bool defineFitter;
  MuonTransientTrackingRecHitBuilder* MuRHBuilder;
  const TransientTrackingRecHitBuilder* TTRHBuilder;

  //                            // LSF for global d(3):    Inf * d = Gfr 
  AlgebraicVector3 Gfr;         // free terms 
  AlgebraicSymMatrix33 Inf;     // information matrix 
  //                            // LSF for global d(6):    Hess * d = Grad 
  CLHEP::HepVector Grad;        // gradient of the objective function in global parameters
  CLHEP::HepMatrix Hess;        // Hessian                  -- // ---     

  CLHEP::HepVector GradL;       // gradient of the objective function in local parameters
  CLHEP::HepMatrix HessL;       // Hessian                  -- // ---     

  int N_event;                  // processed events
  int N_track;                  // selected tracks

  std::vector<AlignTransform>::const_iterator 
  iteratorTrackerRcd;           // location of Tracker in container globalPositionRcd_->m_aligm
  std::vector<AlignTransform>::const_iterator 
  iteratorMuonRcd;              //              Muon
  std::vector<AlignTransform>::const_iterator
  iteratorEcalRcd;              //              Ecal
  std::vector<AlignTransform>::const_iterator
  iteratorHcalRcd;              //              Hcal

  CLHEP::HepVector MuGlShift;   // evaluated global muon shifts  
  CLHEP::HepVector MuGlAngle;   // evaluated global muon angles  

  std::string MuSelect;         // what part of muon system is selected for 1st hit 

  ofstream OutGlobalTxt;        // output the vector of global alignment as text     

  TFile * file;
  TH1F * histo;
  TH1F * histo2; // outerP
  TH1F * histo3; // outerLambda = PI/2-outerTheta
  TH1F * histo4; // phi
  TH1F * histo5; // outerR
  TH1F * histo6; // outerZ
  TH1F * histo7; // s
  TH1F * histo8; // chi^2 of muon-track

  TH1F * histo11; // |Rm-Rt| 
  TH1F * histo12; // Xm-Xt 
  TH1F * histo13; // Ym-Yt 
  TH1F * histo14; // Zm-Zt 
  TH1F * histo15; // Nxm-Nxt 
  TH1F * histo16; // Nym-Nyt 
  TH1F * histo17; // Nzm-Nzt 
  TH1F * histo18; // Error X of inner track  
  TH1F * histo19; // Error X of muon 
  TH1F * histo20; // Error of Xm-Xt 
  TH1F * histo21; // pull(Xm-Xt) 
  TH1F * histo22; // pull(Ym-Yt) 
  TH1F * histo23; // pull(Zm-Zt) 
  TH1F * histo24; // pull(PXm-PXt) 
  TH1F * histo25; // pull(PYm-Pyt) 
  TH1F * histo26; // pull(PZm-PZt) 
  TH1F * histo27; // Nx of tangent plane  
  TH1F * histo28; // Ny of tangent plane  
  TH1F * histo29; // lenght of inner track  
  TH1F * histo30; // lenght of muon track  
  TH1F * histo31; // chi2 local  
  TH1F * histo32; // pull(Pxm/Pzm - Pxt/Pzt) local  
  TH1F * histo33; // pull(Pym/Pzm - Pyt/Pzt) local
  TH1F * histo34; // pull(Xm - Xt) local
  TH1F * histo35; // pull(Ym - Yt) local

  TH2F * histo101; // Rtrack/muon vs Ztrack/muon
  TH2F * histo102; // Ytrack/muon vs Xtrack/muon
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
GlobalTrackerMuonAlignment::GlobalTrackerMuonAlignment(const edm::ParameterSet& iConfig)
  :trackTags_(iConfig.getParameter<edm::InputTag>("tracks")),
   muonTags_(iConfig.getParameter<edm::InputTag>("muons")),
   gmuonTags_(iConfig.getParameter<edm::InputTag>("gmuons")),
   smuonTags_(iConfig.getParameter<edm::InputTag>("smuons")),
   propagator_(iConfig.getParameter<string>("Propagator")),

   cosmicMuonMode_(iConfig.getParameter<bool>("cosmics")),
   isolatedMuonMode_(iConfig.getParameter<bool>("isolated")),

   refitMuon_(iConfig.getParameter<bool>("refitmuon")),
   refitTrack_(iConfig.getParameter<bool>("refittrack")),

   rootOutFile_(iConfig.getUntrackedParameter<string>("rootOutFile")),
   txtOutFile_(iConfig.getUntrackedParameter<string>("txtOutFile")),
   writeDB_(iConfig.getUntrackedParameter<bool>("writeDB")),
   debug_(iConfig.getUntrackedParameter<bool>("debug")),
   defineFitter(true)
{
#ifdef NO_FALSE_FALSE_MODE 
  if (cosmicMuonMode_==false && isolatedMuonMode_==false) {
    throw cms::Exception("BadConfig") << "Muon collection not set! "
				      << "Use from GlobalTrackerMuonAlignment_test_cfg.py.";
  }
#endif  
  if (cosmicMuonMode_==true && isolatedMuonMode_==true) {
    throw cms::Exception("BadConfig") << "Muon collection can be either cosmic or isolated! "
				      << "Please set only one to true.";
  }
}

GlobalTrackerMuonAlignment::~GlobalTrackerMuonAlignment()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

//
// member functions
//

// ------------ method called to for each event  ------------
void
GlobalTrackerMuonAlignment::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{  
  //GlobalTrackerMuonAlignment::analyzeTrackTrack(iEvent, iSetup);  
  GlobalTrackerMuonAlignment::analyzeTrackTrajectory(iEvent, iSetup);  
}

// ------------ method called once for job just before starting event loop  ------------
void GlobalTrackerMuonAlignment::beginJob()
{
  N_event = 0;
  N_track = 0;
  
  if (cosmicMuonMode_==true && isolatedMuonMode_==false) {
    collectionCosmic = true; collectionIsolated = false;}
  else if (cosmicMuonMode_==false && isolatedMuonMode_==true) {
    collectionCosmic = false; collectionIsolated = true;}
  else {
    collectionCosmic = false; collectionIsolated = false;}

  for(int i=0; i<=2; i++){
    Gfr(i) = 0.;
    for(int j=0; j<=2; j++){
      Inf(i,j) = 0.;
    }}

  Grad =  CLHEP::HepVector(6,0);
  Hess =  CLHEP::HepSymMatrix(6,0);

  GradL =  CLHEP::HepVector(6,0);
  HessL =  CLHEP::HepSymMatrix(6,0);

  // histograms
  TDirectory * dirsave = gDirectory;

  file = new TFile(rootOutFile_.c_str(), "recreate"); 
  const bool oldAddDir = TH1::AddDirectoryStatus(); 

  TH1::AddDirectory(true);

  this->bookHist();

  TH1::AddDirectory(oldAddDir);
  dirsave->cd();
}

// ------------ method called once each job just after ending the event loop  ------------
void GlobalTrackerMuonAlignment::endJob() {
  bool alarm = false;

  this->fitHist();

  AlgebraicVector3 d(0., 0., 0.);   // ------------  alignmnet  Global Algebraic

  AlgebraicSymMatrix33 InfI;   // inverse it   
  for(int i=0; i<=2; i++) 
    for(int j=0; j<=2; j++){
      if(j < i) continue;
      InfI(i,j) += Inf(i,j);}
  bool ierr = !InfI.Invert();
  if( ierr ) { if(alarm)
      std::cout<<" Error inverse  Inf matrix !!!!!!!!!!!"<<std::endl;}
  
  for(int i=0; i<=2; i++) 
    for(int k=0; k<=2; k++) d(i) -= InfI(i,k)*Gfr(k);
  // end of Global Algebraic
        
  //                                ---------------  alignment Global CLHEP
  CLHEP::HepVector d3 = CLHEP::solve(Hess, -Grad);
  int iEr3;
  CLHEP::HepMatrix Errd3 = Hess.inverse(iEr3);
  if( iEr3 != 0) { if(alarm)
    std::cout<<" endJob Error inverse Hess matrix !!!!!!!!!!!"<<std::endl;
  }
  // end of Global CLHEP
  
  //                                ----------------- alignment Local CLHEP
  CLHEP::HepVector dLI = CLHEP::solve(HessL, -GradL);
  int iErI;
  CLHEP::HepMatrix ErrdLI = HessL.inverse(iErI);
  if( iErI != 0) { if(alarm)
      std::cout<<" endJob Error inverse HessL matrix !!!!!!!!!!!"<<std::endl;
  }
  // end of Local CLHEP

  // printout of final parameters
  std::cout<<" ---- "<<N_event<<" event "<<N_track<<" tracks "
	   <<MuSelect<<" ---- "<<std::endl;
  if (collectionIsolated) std::cout<<" ALCARECOMuAlCalIsolatedMu "<<std::endl;
  else if (collectionCosmic) std::cout<<"  ALCARECOMuAlGlobalCosmics "<<std::endl; 
  else std::cout<<smuonTags_<<std::endl;
  std::cout<<" Similated shifts[cm] "
	   <<MuGlShift(1)<<" "<<MuGlShift(2)<<" "<<MuGlShift(3)<<" "
	   <<" angles[rad] "<<MuGlAngle(1)<<" "<<MuGlAngle(2)<<" "<<MuGlAngle(3)<<" "
	   <<std::endl;
  std::cout<<" d    "<<-d<<std::endl;;
  std::cout<<" +-   "<<sqrt(InfI(0,0))<<" "<<sqrt(InfI(1,1))<<" "<<sqrt(InfI(2,2))
	   <<std::endl;
  std::cout<<" dG   "<<d3(1)<<" "<<d3(2)<<" "<<d3(3)<<" "<<d3(4)<<" "
	   <<d3(5)<<" "<<d3(6)<<std::endl;;
  std::cout<<" +-   "<<sqrt(Errd3(1,1))<<" "<<sqrt(Errd3(2,2))<<" "<<sqrt(Errd3(3,3))
	   <<" "<<sqrt(Errd3(4,4))<<" "<<sqrt(Errd3(5,5))<<" "<<sqrt(Errd3(6,6))
	   <<std::endl;
  std::cout<<" dL   "<<dLI(1)<<" "<<dLI(2)<<" "<<dLI(3)<<" "<<dLI(4)<<" "
	   <<dLI(5)<<" "<<dLI(6)<<std::endl;;
  std::cout<<" +-   "<<sqrt(ErrdLI(1,1))<<" "<<sqrt(ErrdLI(2,2))<<" "<<sqrt(ErrdLI(3,3))
	   <<" "<<sqrt(ErrdLI(4,4))<<" "<<sqrt(ErrdLI(5,5))<<" "<<sqrt(ErrdLI(6,6))
	   <<std::endl;

  // what do we write to DB
  CLHEP::HepVector vectorToDb(6,0), vectorErrToDb(6,0);
  //vectorToDb = d3;
  //for(unsigned int i=1; i<=6; i++) vectorErrToDb(i) = sqrt(Errd3(i,i));
  vectorToDb = -dLI;
  for(unsigned int i=1; i<=6; i++) vectorErrToDb(i) = sqrt(ErrdLI(i,i));

  // write histograms to root file
  file->Write(); 
  file->Close(); 

  // write global parameters to text file 
  OutGlobalTxt.open(txtOutFile_.c_str(), ios::out);
  if(!OutGlobalTxt.is_open()) std::cout<<" outglobal.txt is not open !!!!!"<<std::endl;
  else{
    OutGlobalTxt<<vectorToDb(1)<<" "<<vectorToDb(2)<<" "<<vectorToDb(3)<<" "<<vectorToDb(4)<<" "
		<<vectorToDb(5)<<" "<<vectorToDb(6)<<" muon Global.\n";
    OutGlobalTxt<<vectorErrToDb(1)<<" "<<vectorErrToDb(1)<<" "<<vectorErrToDb(1)<<" "
		<<vectorErrToDb(1)<<" "<<vectorErrToDb(1)<<" "<<vectorErrToDb(1)<<" errors.\n";
    OutGlobalTxt<<N_event<<" events are processed.\n";

    if (collectionIsolated) OutGlobalTxt<<"ALCARECOMuAlCalIsolatedMu.\n";
    else if (collectionCosmic) OutGlobalTxt<<"  ALCARECOMuAlGlobalCosmics.\n"; 
    else OutGlobalTxt<<smuonTags_<<".\n";
    OutGlobalTxt.close();
    std::cout<<" Write to the file outglobal.txt done  "<<std::endl;
  }

  // write new GlobalPositionRcd to DB
  if(debug_)
    std::cout<<" writeBD_ "<<writeDB_<<std::endl;
  if (writeDB_) 
    GlobalTrackerMuonAlignment::writeGlPosRcd(vectorToDb);
}

// ------------ book histogram  ------------
void GlobalTrackerMuonAlignment::bookHist()
{
  double PI = 3.1415927;
  histo = new TH1F("Pt","pt",1000,0,100); 
  histo2 = new TH1F("P","P [GeV/c]",400,0.,400.);       
  histo2->GetXaxis()->SetTitle("momentum [GeV/c]");
  histo3 = new TH1F("outerLambda","#lambda outer",100,-PI/2.,PI/2.);    
  histo3->GetXaxis()->SetTitle("#lambda outer");
  histo4 = new TH1F("phi","#phi [rad]",100,-PI,PI);       
  histo4->GetXaxis()->SetTitle("#phi [rad]");
  histo5 = new TH1F("Rmuon","inner muon hit R [cm]",100,0.,800.);       
  histo5->GetXaxis()->SetTitle("R of muon [cm]");
  histo6 = new TH1F("Zmuon","inner muon hit Z[cm]",100,-1000.,1000.);       
  histo6->GetXaxis()->SetTitle("Z of muon [cm]");
  histo7 = new TH1F("(Pm-Pt)/Pt"," (Pmuon-Ptrack)/Ptrack", 100,-2.,2.);       
  histo7->GetXaxis()->SetTitle("(Pmuon-Ptrack)/Ptrack");
  histo8 = new TH1F("chi muon-track","#chi^{2}(muon-track)", 1000,0.,1000.);       
  histo8->GetXaxis()->SetTitle("#chi^{2} of muon w.r.t. propagated track");
  histo11 = new TH1F("distance muon-track","distance muon w.r.t track [cm]",100,0.,30.);
  histo11->GetXaxis()->SetTitle("distance of muon w.r.t. track [cm]");
  histo12 = new TH1F("Xmuon-Xtrack","Xmuon-Xtrack [cm]",200,-20.,20.);
  histo12->GetXaxis()->SetTitle("Xmuon - Xtrack [cm]"); 
  histo13 = new TH1F("Ymuon-Ytrack","Ymuon-Ytrack [cm]",200,-20.,20.);
  histo13->GetXaxis()->SetTitle("Ymuon - Ytrack [cm]");
  histo14 = new TH1F("Zmuon-Ztrack","Zmuon-Ztrack [cm]",200,-20.,20.);       
  histo14->GetXaxis()->SetTitle("Zmuon-Ztrack [cm]");
  histo15 = new TH1F("NXmuon-NXtrack","NXmuon-NXtrack [rad]",200,-.1,.1);       
  histo15->GetXaxis()->SetTitle("N_{X}(muon)-N_{X}(track) [rad]");
  histo16 = new TH1F("NYmuon-NYtrack","NYmuon-NYtrack [rad]",200,-.1,.1);       
  histo16->GetXaxis()->SetTitle("N_{Y}(muon)-N_{Y}(track) [rad]");
  histo17 = new TH1F("NZmuon-NZtrack","NZmuon-NZtrack [rad]",200,-.1,.1);       
  histo17->GetXaxis()->SetTitle("N_{Z}(muon)-N_{Z}(track) [rad]");
  histo18 = new TH1F("expected error of Xinner","outer hit of inner tracker",100,0,.01);
  histo18->GetXaxis()->SetTitle("expected error of Xinner [cm]");
  histo19 = new TH1F("expected error of Xmuon","inner hit of muon",100,0,.1);       
  histo19->GetXaxis()->SetTitle("expected error of Xmuon [cm]");
  histo20 = new TH1F("expected error of Xmuon-Xtrack","muon w.r.t. propagated track",100,0.,10.);
  histo20->GetXaxis()->SetTitle("expected error of Xmuon-Xtrack [cm]");
  histo21 = new TH1F("pull of Xmuon-Xtrack","pull of Xmuon-Xtrack",100,-10.,10.);
  histo21->GetXaxis()->SetTitle("(Xmuon-Xtrack)/expected error");
  histo22 = new TH1F("pull of Ymuon-Ytrack","pull of Ymuon-Ytrack",100,-10.,10.);       
  histo22->GetXaxis()->SetTitle("(Ymuon-Ytrack)/expected error");
  histo23 = new TH1F("pull of Zmuon-Ztrack","pull of Zmuon-Ztrack",100,-10.,10.);       
  histo23->GetXaxis()->SetTitle("(Zmuon-Ztrack)/expected error");
  histo24 = new TH1F("pull of PXmuon-PXtrack","pull of PXmuon-PXtrack",100,-10.,10.);       
  histo24->GetXaxis()->SetTitle("(P_{X}(muon)-P_{X}(track))/expected error");
  histo25 = new TH1F("pull of PYmuon-PYtrack","pull of PYmuon-PYtrack",100,-10.,10.);       
  histo25->GetXaxis()->SetTitle("(P_{Y}(muon)-P_{Y}(track))/expected error");
  histo26 = new TH1F("pull of PZmuon-PZtrack","pull of PZmuon-PZtrack",100,-10.,10.);       
  histo26->GetXaxis()->SetTitle("(P_{Z}(muon)-P_{Z}(track))/expected error");
  histo27 = new TH1F("N_x","Nx of tangent plane",120,-1.1,1.1); 
  histo27->GetXaxis()->SetTitle("normal vector projection N_{X}");
  histo28 = new TH1F("N_y","Ny of tangent plane",120,-1.1,1.1); 
  histo28->GetXaxis()->SetTitle("normal vector projection N_{Y}");
  histo29 = new TH1F("lenght of track","lenght of track",200,0.,400); 
  histo29->GetXaxis()->SetTitle("lenght of track [cm]");
  histo30 = new TH1F("lenght of muon","lenght of muon",200,0.,800); 
  histo30->GetXaxis()->SetTitle("lenght of muon [cm]");

  histo31 = new TH1F("local chi muon-track","#local chi^{2}(muon-track)", 1000,0.,1000.);       
  histo31->GetXaxis()->SetTitle("#local chi^{2} of muon w.r.t. propagated track");
  histo32 = new TH1F("pull of Px/Pz local","pull of Px/Pz local",100,-10.,10.);
  histo32->GetXaxis()->SetTitle("local (Px/Pz(muon) - Px/Pz(track))/expected error");
  histo33 = new TH1F("pull of Py/Pz local","pull of Py/Pz local",100,-10.,10.);
  histo33->GetXaxis()->SetTitle("local (Py/Pz(muon) - Py/Pz(track))/expected error");
  histo34 = new TH1F("pull of X local","pull of X local",100,-10.,10.);
  histo34->GetXaxis()->SetTitle("local (Xmuon - Xtrack)/expected error");
  histo35 = new TH1F("pull of Y local","pull of Y local",100,-10.,10.);
  histo35->GetXaxis()->SetTitle("local (Ymuon - Ytrack)/expected error");

  histo101 = new TH2F("Rtr/mu vs Ztr/mu","hit of track/muon",100,-800.,800.,100,0.,600.);
  histo101->GetXaxis()->SetTitle("Z of track/muon [cm]");
  histo101->GetYaxis()->SetTitle("R of track/muon [cm]");
  histo102 = new TH2F("Ytr/mu vs Xtr/mu","hit of track/muon",100,-600.,600.,100,-600.,600.);
  histo102->GetXaxis()->SetTitle("X of track/muon [cm]");
  histo102->GetYaxis()->SetTitle("Y of track/muon [cm]");
}

// ------------ fit histogram  ------------
void GlobalTrackerMuonAlignment::fitHist() {

  histo7->Fit("gaus","Q");

  histo12->Fit("gaus","Q");
  histo13->Fit("gaus","Q");
  histo14->Fit("gaus","Q");
  histo15->Fit("gaus","Q");
  histo16->Fit("gaus","Q");
  histo17->Fit("gaus","Q");

  histo21->Fit("gaus","Q");
  histo22->Fit("gaus","Q");
  histo23->Fit("gaus","Q");
  histo24->Fit("gaus","Q");
  histo25->Fit("gaus","Q");
  histo26->Fit("gaus","Q");

  histo32->Fit("gaus","Q");
  histo33->Fit("gaus","Q");
  histo34->Fit("gaus","Q");
  histo35->Fit("gaus","Q");
}

// ------------ method to analyze recoTrack & recoMuon of Global Muon  ------------
void GlobalTrackerMuonAlignment::analyzeTrackTrack
(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;  
  //debug_ = true;
  bool info = false;
  bool alarm = false;
  //bool alarm = true;
  double PI = 3.1415927;
 
  cosmicMuonMode_ = true; // true: both Cosmic and IsolatedMuon are treated with 1,2 tracks
  isolatedMuonMode_ = !cosmicMuonMode_; //true: only IsolatedMuon are treated with 1 track
 
  //if(N_event == 1)
  //GlobalPositionRcdRead1::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  N_event++;
  if (info || debug_) {
    std::cout << "----- event " << N_event << " -- tracks "<<N_track<<" ---";
    if (cosmicMuonMode_) std::cout << " treated as CosmicMu ";
    if (isolatedMuonMode_) std::cout <<" treated as IsolatedMu ";
    std::cout << std::endl;
  }

  Handle<reco::TrackCollection> tracks;
  Handle<reco::TrackCollection> muons;
  Handle<reco::TrackCollection> gmuons;
  Handle<reco::MuonCollection> smuons;

  if (collectionIsolated){
    iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "TrackerOnly",  tracks);
    iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "StandAlone",  muons);
    iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "GlobalMuon",  gmuons);
    iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "SelectedMuons",  smuons);
  }
  else if (collectionCosmic){
    iEvent.getByLabel("ALCARECOMuAlGlobalCosmics", "TrackerOnly",  tracks);
    iEvent.getByLabel("ALCARECOMuAlGlobalCosmics", "StandAlone",  muons);
    iEvent.getByLabel("ALCARECOMuAlGlobalCosmics", "GlobalMuon",  gmuons);
    iEvent.getByLabel("ALCARECOMuAlGlobalCosmics", "SelectedMuons",  smuons);
  }
  else{
    iEvent.getByLabel(trackTags_,tracks);
    iEvent.getByLabel(muonTags_,muons);
    iEvent.getByLabel(gmuonTags_,gmuons);
    iEvent.getByLabel(smuonTags_,smuons);
  }  

  if (debug_) {
    std::cout << " ievBunch " << iEvent.bunchCrossing()
	      << " runN " << (int)iEvent.run() <<std::endl;
    std::cout << " N tracks s/amu gmu selmu "<<tracks->size() << " " <<muons->size()
              << " "<<gmuons->size() << " " << smuons->size() << std::endl;
    for (MuonCollection::const_iterator itMuon = smuons->begin();
	itMuon != smuons->end();
	 ++itMuon) {
      std::cout << " is isolatValid Matches " << itMuon->isIsolationValid()
		<< " " <<itMuon->isMatchesValid() << std::endl;
    }
  }

  if (isolatedMuonMode_) {                // ---- Only 1 Isolated Muon --------
    if (tracks->size() != 1) return;  
    if (muons->size()  != 1) return;  
    if (gmuons->size() != 1) return; 
    if (smuons->size() != 1) return; 
  }
  
  if (cosmicMuonMode_){                // ---- 1,2 Cosmic Muon --------
    if (smuons->size() > 2) return; 
    if (tracks->size() != smuons->size()) return;  
    if (muons->size() != smuons->size()) return;  
    if (gmuons->size() != smuons->size()) return;  
  }
  
  // ok mc_isolated_mu
  //edm::ESHandle<TrackerGeometry> trackerGeometry;
  //iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);
  // ok mc_isolated_mu
  //edm::ESHandle<DTGeometry> dtGeometry;
  //iSetup.get<MuonGeometryRecord>().get(dtGeometry);
  // don't work
  //edm::ESHandle<CSCGeometry> cscGeometry;
  //iSetup.get<MuonGeometryRecord>().get(cscGeometry);
  
  if (watchTrackingGeometry_.check(iSetup) || !trackingGeometry_) {
    edm::ESHandle<GlobalTrackingGeometry> trackingGeometry;
    iSetup.get<GlobalTrackingGeometryRecord>().get(trackingGeometry);
    trackingGeometry_ = &*trackingGeometry;
  }
  
  if (watchMagneticFieldRecord_.check(iSetup) || !magneticField_) {
    edm::ESHandle<MagneticField> magneticField;
    iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
    magneticField_ = &*magneticField;
  }

  if (watchGlobalPositionRcd_.check(iSetup) || !globalPositionRcd_) {
    edm::ESHandle<Alignments> globalPositionRcd;
    iSetup.get<GlobalPositionRcd>().get(globalPositionRcd);
    globalPositionRcd_ = &*globalPositionRcd;
    for (std::vector<AlignTransform>::const_iterator i 
	   = globalPositionRcd_->m_align.begin();
	 i != globalPositionRcd_->m_align.end();  ++i){
      if(DetId(DetId::Tracker).rawId() == i->rawId()) iteratorTrackerRcd = i;
      if(DetId(DetId::Muon).rawId() == i->rawId()) iteratorMuonRcd = i;
      if(DetId(DetId::Ecal).rawId() == i->rawId()) iteratorEcalRcd = i;
      if(DetId(DetId::Hcal).rawId() == i->rawId()) iteratorHcalRcd = i;
    }
    if(true || debug_){
      std::cout << "=== iteratorMuonRcd " << iteratorMuonRcd->rawId()<<" ====\n" 
		<< " translation " << iteratorMuonRcd->translation()<<"\n" 
		<< " angles " << iteratorMuonRcd->rotation().eulerAngles() << std::endl;
      std::cout << iteratorMuonRcd->rotation() << std::endl;
    }   
  } // end of GlobalPositionRcd
  
  ESHandle<Propagator> propagator;
  iSetup.get<TrackingComponentsRecord>().get(propagator_, propagator);
  
  SteppingHelixPropagator alongStHePr = 
    SteppingHelixPropagator(magneticField_, alongMomentum);  
  SteppingHelixPropagator oppositeStHePr = 
    SteppingHelixPropagator(magneticField_, oppositeToMomentum);  

  //double tolerance = 5.e-5;
  defaultRKPropagator::Product  aprod( magneticField_, alongMomentum, 5.e-5); auto & alongRKPr = aprod.propagator;
  defaultRKPropagator::Product  oprod( magneticField_, oppositeToMomentum, 5.e-5); auto & oppositeRKPr = oprod.propagator;

  float epsilon = 5.;
  SmartPropagator alongSmPr = 
    SmartPropagator(alongRKPr, alongStHePr, magneticField_, alongMomentum, epsilon);
  SmartPropagator oppositeSmPr = 
    SmartPropagator(oppositeRKPr, oppositeStHePr, magneticField_, oppositeToMomentum, epsilon);
  
  // ................................................ selected/global muon
  //itMuon -->  Jim's globalMuon  
  for(MuonCollection::const_iterator itMuon = smuons->begin();
      itMuon != smuons->end(); ++itMuon) {
    
    if(debug_){
      std::cout<<" mu gM is GM Mu SaM tM "<<itMuon->isGlobalMuon()<<" "
	       <<itMuon->isMuon()<<" "<<itMuon->isStandAloneMuon()
	       <<" "<<itMuon->isTrackerMuon()<<" "
	       <<std::endl;
    }
    
    // get information about the innermost muon hit -------------------------
    TransientTrack muTT(itMuon->outerTrack(), magneticField_, trackingGeometry_);
    TrajectoryStateOnSurface innerMuTSOS = muTT.innermostMeasurementState();
    TrajectoryStateOnSurface outerMuTSOS = muTT.outermostMeasurementState();
    
    // get information about the outermost tracker hit -----------------------
    TransientTrack trackTT(itMuon->track(), magneticField_, trackingGeometry_);
    TrajectoryStateOnSurface outerTrackTSOS = trackTT.outermostMeasurementState();
    TrajectoryStateOnSurface innerTrackTSOS = trackTT.innermostMeasurementState();
    
    GlobalPoint pointTrackIn  = innerTrackTSOS.globalPosition();
    GlobalPoint pointTrackOut = outerTrackTSOS.globalPosition();
    float lenghtTrack = (pointTrackOut-pointTrackIn).mag();
    GlobalPoint pointMuonIn  = innerMuTSOS.globalPosition();
    GlobalPoint pointMuonOut = outerMuTSOS.globalPosition();
    float lenghtMuon = (pointMuonOut - pointMuonIn).mag();
    GlobalVector momentumTrackOut = outerTrackTSOS.globalMomentum();
    GlobalVector momentumTrackIn = innerTrackTSOS.globalMomentum();
    float distanceInIn   = (pointTrackIn  - pointMuonIn).mag();
    float distanceInOut  = (pointTrackIn  - pointMuonOut).mag();
    float distanceOutIn  = (pointTrackOut - pointMuonIn).mag();
    float distanceOutOut = (pointTrackOut - pointMuonOut).mag();

    if(debug_){
      std::cout<<" pointTrackIn "<<pointTrackIn<<std::endl;
      std::cout<<"          Out "<<pointTrackOut<<" lenght "<<lenghtTrack<<std::endl;
      std::cout<<" point MuonIn "<<pointMuonIn<<std::endl;
      std::cout<<"          Out "<<pointMuonOut<<" lenght "<<lenghtMuon<<std::endl;
      std::cout<<" momeTrackIn Out "<<momentumTrackIn<<" "<<momentumTrackOut<<std::endl;
      std::cout<<" doi io ii oo "<<distanceOutIn<<" "<<distanceInOut<<" "
	       <<distanceInIn<<" "<<distanceOutOut<<std::endl; 
  }   

    if(lenghtTrack < 90.) continue;
    if(lenghtMuon < 300.) continue;
    if(momentumTrackIn.mag() < 15. || momentumTrackOut.mag() < 15.) continue;
    if(trackTT.charge() != muTT.charge()) continue;

    if(debug_)
      std::cout<<" passed lenght momentum cuts"<<std::endl;

    GlobalVector GRm, GPm, Nl, Rm, Pm, Rt, Pt, Rt0;
    AlgebraicSymMatrix66 Cm, C0, Ce, C1;

    TrajectoryStateOnSurface extrapolationT;
    int tsosMuonIf = 0;

    if( isolatedMuonMode_ ){      //------------------------------- Isolated Muon -----
      const Surface& refSurface = innerMuTSOS.surface(); 
      ConstReferenceCountingPointer<TangentPlane> 
	tpMuLocal(refSurface.tangentPlane(innerMuTSOS.localPosition()));
      Nl = tpMuLocal->normalVector();
      ConstReferenceCountingPointer<TangentPlane> 
      tpMuGlobal(refSurface.tangentPlane(innerMuTSOS.globalPosition()));
      GlobalVector Ng = tpMuGlobal->normalVector();
      Surface* surf = (Surface*)&refSurface;
      const Plane* refPlane = dynamic_cast<Plane*>(surf); 
      GlobalVector Nlp = refPlane->normalVector();
      
      if(debug_){
	std::cout<<" Nlocal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
	 <<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
	std::cout<<"  global "<<Ng.x()<<" "<<Ng.y()<<" "<<Ng.z()<<std::endl;
	std::cout<<"      lp "<<Nlp.x()<<" "<<Nlp.y()<<" "<<Nlp.z()<<std::endl;
	//std::cout<<" Nlocal Nglobal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
	// <<Ng.x()<<" "<<Ng.y()<<" "<<Ng.z()
	//<<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
      }

      //                          extrapolation to innermost muon hit     
      extrapolationT = alongSmPr.propagate(outerTrackTSOS, refSurface);
      //extrapolationT = propagator->propagate(outerTrackTSOS, refSurface);
      
      if(!extrapolationT.isValid()){
	if(false & alarm)
	  std::cout<<" !!!!!!!!! Catastrophe Out-In extrapolationT.isValid "
	    //<<"\a\a\a\a\a\a\a\a"<<extrapolationT.isValid()
		   <<std::endl;
	continue;
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
      Cm = AlgebraicSymMatrix66 (extrapolationT.cartesianError().matrix() + 
				 innerMuTSOS.cartesianError().matrix());
      C0 = AlgebraicSymMatrix66(outerTrackTSOS.cartesianError().matrix()); 		   
      Ce = AlgebraicSymMatrix66(extrapolationT.cartesianError().matrix()); 
      C1 = AlgebraicSymMatrix66(innerMuTSOS.cartesianError().matrix());
     
    }  //                    -------------------------------   end Isolated Muon -----
    
    
    if( cosmicMuonMode_ ){      //------------------------------- Cosmic Muon -----

      if((distanceOutIn <= distanceInOut) & (distanceOutIn <= distanceInIn) & 
	 (distanceOutIn <= distanceOutOut)){             // ----- Out - In ------    
	if(debug_) std::cout<<" -----    Out - In -----"<<std::endl;

        const Surface& refSurface = innerMuTSOS.surface(); 
	ConstReferenceCountingPointer<TangentPlane> 
	  tpMuGlobal(refSurface.tangentPlane(innerMuTSOS.globalPosition()));
	Nl = tpMuGlobal->normalVector();

	//                          extrapolation to innermost muon hit     
	extrapolationT = alongSmPr.propagate(outerTrackTSOS, refSurface);
	//extrapolationT = propagator->propagate(outerTrackTSOS, refSurface);

	if(!extrapolationT.isValid()){
	  if(false & alarm)
	    std::cout<<" !!!!!!!!! Catastrophe Out-In extrapolationT.isValid "
	      //<<"\a\a\a\a\a\a\a\a"<<extrapolationT.isValid()
		     <<std::endl;
	  continue;
	}
	if(debug_)
	  std::cout<<" extrapolationT.isValid "<<extrapolationT.isValid()<<std::endl; 

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
	Cm = AlgebraicSymMatrix66 (extrapolationT.cartesianError().matrix() + 
				   innerMuTSOS.cartesianError().matrix());
	C0 = AlgebraicSymMatrix66(outerTrackTSOS.cartesianError().matrix()); 		   
	Ce = AlgebraicSymMatrix66(extrapolationT.cartesianError().matrix()); 
	C1 = AlgebraicSymMatrix66(innerMuTSOS.cartesianError().matrix());
	
	if(false & debug_){
	  //std::cout<<" ->propDir "<<propagator->propagationDirection()<<std::endl;
	  PropagationDirectionChooser Chooser;
	  std::cout<<" propDirCh "
		   <<Chooser.operator()(*outerTrackTSOS.freeState(), refSurface)
		   <<" Ch == along "<<(alongMomentum == 
				       Chooser.operator()(*outerTrackTSOS.freeState(), refSurface))
		   <<std::endl;
	  std::cout<<" --- Nlocal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
		   <<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
	}
      }           //                                       enf of ---- Out - In -----
      

      else if((distanceInOut <= distanceInIn) & (distanceInOut <= distanceOutIn) & 
	      (distanceInOut <= distanceOutOut)){            // ----- In - Out ------    
	if(debug_) std::cout<<" -----    In - Out -----"<<std::endl;

	const Surface& refSurface = outerMuTSOS.surface(); 
	ConstReferenceCountingPointer<TangentPlane> 
	  tpMuGlobal(refSurface.tangentPlane(outerMuTSOS.globalPosition()));
	Nl = tpMuGlobal->normalVector();
		
	//                          extrapolation to outermost muon hit     
	extrapolationT = oppositeSmPr.propagate(innerTrackTSOS, refSurface);

	if(!extrapolationT.isValid()){
	  if(false & alarm)
	    std::cout<<" !!!!!!!!! Catastrophe Out-In extrapolationT.isValid "
	      <<"\a\a\a\a\a\a\a\a"<<extrapolationT.isValid()
		     <<std::endl;
	  continue;
	}
	if(debug_)
	  std::cout<<" extrapolationT.isValid "<<extrapolationT.isValid()<<std::endl; 
	
	tsosMuonIf = 2;
	Rt = GlobalVector((extrapolationT.globalPosition()).x(),
			  (extrapolationT.globalPosition()).y(),
			  (extrapolationT.globalPosition()).z());
	
	Pt = extrapolationT.globalMomentum();
	//                          global parameters of muon
	GRm = GlobalVector((outerMuTSOS.globalPosition()).x(),
			   (outerMuTSOS.globalPosition()).y(),
			   (outerMuTSOS.globalPosition()).z());
	GPm = outerMuTSOS.globalMomentum();
	Rt0 = GlobalVector((innerTrackTSOS.globalPosition()).x(),
			   (innerTrackTSOS.globalPosition()).y(),
			   (innerTrackTSOS.globalPosition()).z());
	Cm = AlgebraicSymMatrix66 (extrapolationT.cartesianError().matrix() + 
				   outerMuTSOS.cartesianError().matrix());
	C0 = AlgebraicSymMatrix66(innerTrackTSOS.cartesianError().matrix()); 		   
	Ce = AlgebraicSymMatrix66(extrapolationT.cartesianError().matrix()); 
	C1 = AlgebraicSymMatrix66(outerMuTSOS.cartesianError().matrix());
	
	if(false & debug_){
	  //std::cout<<" ->propDir "<<propagator->propagationDirection()<<std::endl;
	  PropagationDirectionChooser Chooser;
	  std::cout<<" propDirCh "
		   <<Chooser.operator()(*innerTrackTSOS.freeState(), refSurface)
		   <<" Ch == oppisite "<<(oppositeToMomentum == 
					  Chooser.operator()(*innerTrackTSOS.freeState(), refSurface))
		   <<std::endl;
	  std::cout<<" --- Nlocal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
		   <<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
	}
      }           //                                        enf of ---- In - Out -----

      else if((distanceOutOut <= distanceInOut) & (distanceOutOut <= distanceInIn) & 
	      (distanceOutOut <= distanceOutIn)){          // ----- Out - Out ------    
	if(debug_) std::cout<<" -----    Out - Out -----"<<std::endl;

	// reject: momentum of track has opposite direction to muon track
	continue;

	const Surface& refSurface = outerMuTSOS.surface(); 
	ConstReferenceCountingPointer<TangentPlane> 
	  tpMuGlobal(refSurface.tangentPlane(outerMuTSOS.globalPosition()));
	Nl = tpMuGlobal->normalVector();
		
	//                          extrapolation to outermost muon hit     
	extrapolationT = alongSmPr.propagate(outerTrackTSOS, refSurface);

	if(!extrapolationT.isValid()){
	  if(alarm)
	    std::cout<<" !!!!!!!!! Catastrophe Out-Out extrapolationT.isValid "
		     <<"\a\a\a\a\a\a\a\a"<<extrapolationT.isValid()
		     <<std::endl;
	  continue;
	}
	if(debug_)
	  std::cout<<" extrapolationT.isValid "<<extrapolationT.isValid()<<std::endl; 

	tsosMuonIf = 2;
	Rt = GlobalVector((extrapolationT.globalPosition()).x(),
			  (extrapolationT.globalPosition()).y(),
			  (extrapolationT.globalPosition()).z());
	
	Pt = extrapolationT.globalMomentum();
	//                          global parameters of muon
	GRm = GlobalVector((outerMuTSOS.globalPosition()).x(),
			   (outerMuTSOS.globalPosition()).y(),
			   (outerMuTSOS.globalPosition()).z());
	GPm = outerMuTSOS.globalMomentum();
	Rt0 = GlobalVector((outerTrackTSOS.globalPosition()).x(),
			   (outerTrackTSOS.globalPosition()).y(),
			   (outerTrackTSOS.globalPosition()).z());
	Cm = AlgebraicSymMatrix66 (extrapolationT.cartesianError().matrix() + 
				   outerMuTSOS.cartesianError().matrix());
	C0 = AlgebraicSymMatrix66(outerTrackTSOS.cartesianError().matrix()); 		   
	Ce = AlgebraicSymMatrix66(extrapolationT.cartesianError().matrix()); 
	C1 = AlgebraicSymMatrix66(outerMuTSOS.cartesianError().matrix());
	
	if(debug_){
	  PropagationDirectionChooser Chooser;
	  std::cout<<" propDirCh "
		   <<Chooser.operator()(*outerTrackTSOS.freeState(), refSurface)
		   <<" Ch == along "<<(alongMomentum == 
				       Chooser.operator()(*outerTrackTSOS.freeState(), refSurface))
		   <<std::endl;
	  std::cout<<" --- Nlocal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
		   <<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
	  std::cout<<"     Nornal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
		   <<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
	}
      }           //                                       enf of ---- Out - Out -----
      else{
	if(alarm)
	  std::cout<<" ------- !!!!!!!!!!!!!!! No proper Out - In \a\a\a\a\a\a\a"<<std::endl; 
	continue;
      }
      
    }  //                     -------------------------------   end Cosmic Muon -----
    
    if(tsosMuonIf == 0) {if(info) {std::cout<<"No tsosMuon !!!!!!"<<std::endl; continue;}}
    TrajectoryStateOnSurface tsosMuon;
    if(tsosMuonIf == 1) tsosMuon = muTT.innermostMeasurementState();
    else tsosMuon = muTT.outermostMeasurementState();

    //GlobalTrackerMuonAlignment::misalignMuon(GRm, GPm, Nl, Rt, Rm, Pm);
    AlgebraicVector4 LPRm;  // muon local (dx/dz, dy/dz, x, y)
    GlobalTrackerMuonAlignment::misalignMuonL
    (GRm, GPm, Nl, Rt, Rm, Pm, LPRm, extrapolationT, tsosMuon);

    GlobalVector resR = Rm - Rt;
    GlobalVector resP0 = Pm - Pt;
    GlobalVector resP = Pm / Pm.mag() - Pt / Pt.mag();
    float RelMomResidual = (Pm.mag() - Pt.mag()) / (Pt.mag() + 1.e-6);;

    AlgebraicVector6 Vm;
    Vm(0) = resR.x();  Vm(1) = resR.y();  Vm(2) = resR.z(); 
    Vm(3) = resP0.x();  Vm(4) = resP0.y();  Vm(5) = resP0.z(); 
    float Rmuon = Rm.perp();
    float Zmuon = Rm.z();
    float alfa_x = atan2(Nl.x(),Nl.y())*57.29578;
    
    if(debug_){
      std::cout<<" Nx Ny Nz alfa_x "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "<<alfa_x<<std::endl;
      //std::cout<<" Rm "<<Rm<<std::endl<<" Rt "<<Rt<<std::endl<<" resR "<<resR<<std::endl
      //       <<" resP "<<resP<<" dp/p "<<RelMomResidual<<std::endl;
    }

    double chi_d = 0;
    for(int i=0; i<=5; i++) chi_d += Vm(i)*Vm(i)/Cm(i,i);
        
    AlgebraicVector5 Vml(tsosMuon.localParameters().vector() 
		       - extrapolationT.localParameters().vector());
    AlgebraicSymMatrix55 m(tsosMuon.localError().matrix() + extrapolationT.localError().matrix());
    AlgebraicSymMatrix55 Cml(tsosMuon.localError().matrix() + extrapolationT.localError().matrix());
    bool ierrLoc = !m.Invert();
    if (ierrLoc && debug_ && info) { 
      std::cout<< " ==== Error inverting Local covariance matrix ==== "<<std::cout;
      continue;}
    double chi_Loc = ROOT::Math::Similarity(Vml,m);
    if(debug_)
      std::cout<<" chi_Loc px/pz/err "<<chi_Loc<<" "<<Vml(1)/sqrt(Cml(1,1))<<std::endl;

    if(Pt.mag() < 15.) continue;
    if(Pm.mag() <  5.) continue;
 
    //if(Pt.mag() < 30.) continue;          // momenum cut  < 30GeV
    //if(Pt.mag() < 60.) continue;          // momenum cut  < 30GeV
    //if(Pt.mag() > 50.) continue;          //  momenum cut > 50GeV
    //if(Pt.mag() > 100.) continue;          //  momenum cut > 100GeV
    //if(trackTT.charge() < 0) continue;    // select positive charge
    //if(trackTT.charge() > 0) continue;    // select negative charge
    
    //if(fabs(resR.x()) > 5.) continue;     // strong cut  X 
    //if(fabs(resR.y()) > 5.) continue;     //             Y 
    //if(fabs(resR.z()) > 5.) continue;     //             Z
    //if(fabs(resR.mag()) > 7.5) continue;   //            dR

    /*
    //if(fabs(RelMomResidual) > 0.5) continue;
    if(fabs(resR.x()) > 20.) continue;
    if(fabs(resR.y()) > 20.) continue;
    if(fabs(resR.z()) > 20.) continue;
    if(fabs(resR.mag()) > 30.) continue;
    if(fabs(resP.x()) > 0.06) continue;
    if(fabs(resP.y()) > 0.06) continue;
    if(fabs(resP.z()) > 0.06) continue;
    if(chi_d > 40.) continue;
    */

    //                                            select Barrel 
    //if(Rmuon < 400. || Rmuon > 450.) continue; 
    //if(Zmuon < -600. || Zmuon > 600.) continue;
    //if(fabs(Nl.z()) > 0.95) continue;
    //MuSelect = " Barrel";
    //                                                  EndCap1
    //if(Rmuon < 120. || Rmuon > 450.) continue;
    //if(Zmuon < -720.) continue;
    //if(Zmuon > -580.) continue;
    //if(fabs(Nl.z()) < 0.95) continue;  
    //MuSelect = " EndCap1";
    //                                                  EndCap2
    //if(Rmuon < 120. || Rmuon > 450.) continue;
    //if(Zmuon >  720.) continue;
    //if(Zmuon <  580.) continue;
    //if(fabs(Nl.z()) < 0.95) continue;  
    //MuSelect = " EndCap2";
    //                                                 select All
    if(Rmuon < 120. || Rmuon > 450.) continue;  
    if(Zmuon < -720. || Zmuon > 720.) continue;
    MuSelect = " Barrel+EndCaps";

    
    if(debug_)
      std::cout<<" .............. passed all cuts"<<std::endl;
        
    N_track++;
    //                     gradient and Hessian for each track
    
    GlobalTrackerMuonAlignment::gradientGlobalAlg(Rt, Pt, Rm, Nl, Cm);
    GlobalTrackerMuonAlignment::gradientGlobal(Rt, Pt, Rm, Pm, Nl, Cm); 

    CLHEP::HepSymMatrix covLoc(4,0);
    for(int i=1; i<=4; i++)
      for(int j=1; j<=i; j++){
	covLoc(i,j) = (tsosMuon.localError().matrix()
		       + extrapolationT.localError().matrix())(i,j);
	//if(i != j) Cov(i,j) = 0.;
      }

    const Surface& refSurface = tsosMuon.surface(); 
    CLHEP::HepMatrix rotLoc (3,3,0);
    rotLoc(1,1) = refSurface.rotation().xx();
    rotLoc(1,2) = refSurface.rotation().xy();
    rotLoc(1,3) = refSurface.rotation().xz();
    
    rotLoc(2,1) = refSurface.rotation().yx();
    rotLoc(2,2) = refSurface.rotation().yy();
    rotLoc(2,3) = refSurface.rotation().yz();
    
    rotLoc(3,1) = refSurface.rotation().zx();
    rotLoc(3,2) = refSurface.rotation().zy();
    rotLoc(3,3) = refSurface.rotation().zz();
    
    CLHEP::HepVector posLoc(3);
    posLoc(1) = refSurface.position().x();
    posLoc(2) = refSurface.position().y();
    posLoc(3) = refSurface.position().z();

    GlobalTrackerMuonAlignment::gradientLocal(Rt, Pt, Rm, Pm, Nl, covLoc, rotLoc, posLoc, LPRm);

    if(debug_){
      std::cout<<" Norm   "<<Nl<<std::endl;
      std::cout<<" posLoc "<<posLoc.T()<<std::endl;
      std::cout<<" rotLoc "<<rotLoc<<std::endl;
    }

    // -----------------------------------------------------  fill histogram 
    histo->Fill(itMuon->track()->pt()); 
    
    //histo2->Fill(itMuon->track()->outerP()); 
    histo2->Fill(Pt.mag()); 
    histo3->Fill((PI/2.-itMuon->track()->outerTheta())); 
    histo4->Fill(itMuon->track()->phi()); 
    histo5->Fill(Rmuon); 
    histo6->Fill(Zmuon); 
    histo7->Fill(RelMomResidual); 
    //histo8->Fill(chi); 
    histo8->Fill(chi_d); 
    
    histo101->Fill(Zmuon, Rmuon); 
    histo101->Fill(Rt0.z(), Rt0.perp()); 
    histo102->Fill(Rt0.x(), Rt0.y()); 
    histo102->Fill(Rm.x(), Rm.y()); 
    
    histo11->Fill(resR.mag()); 
    if(fabs(Nl.x()) < 0.98) histo12->Fill(resR.x()); 
    if(fabs(Nl.y()) < 0.98) histo13->Fill(resR.y()); 
    if(fabs(Nl.z()) < 0.98) histo14->Fill(resR.z()); 
    histo15->Fill(resP.x()); 
    histo16->Fill(resP.y()); 
    histo17->Fill(resP.z()); 
    
    if((fabs(Nl.x()) < 0.98) && (fabs(Nl.y()) < 0.98) &&(fabs(Nl.z()) < 0.98))
      { 
	histo18->Fill(sqrt(C0(0,0))); 
	histo19->Fill(sqrt(C1(0,0))); 
	histo20->Fill(sqrt(C1(0,0)+Ce(0,0))); 		   
      }
    if(fabs(Nl.x()) < 0.98) histo21->Fill(Vm(0)/sqrt(Cm(0,0))); 
    if(fabs(Nl.y()) < 0.98) histo22->Fill(Vm(1)/sqrt(Cm(1,1))); 
    if(fabs(Nl.z()) < 0.98) histo23->Fill(Vm(2)/sqrt(Cm(2,2))); 
    histo24->Fill(Vm(3)/sqrt(C1(3,3)+Ce(3,3))); 
    histo25->Fill(Vm(4)/sqrt(C1(4,4)+Ce(4,4))); 
    histo26->Fill(Vm(5)/sqrt(C1(5,5)+Ce(5,5))); 
    histo27->Fill(Nl.x()); 
    histo28->Fill(Nl.y()); 
    histo29->Fill(lenghtTrack); 
    histo30->Fill(lenghtMuon);     
    histo31->Fill(chi_Loc);
    histo32->Fill(Vml(1)/sqrt(Cml(1,1))); 
    histo33->Fill(Vml(2)/sqrt(Cml(2,2))); 
    histo34->Fill(Vml(3)/sqrt(Cml(3,3))); 
    histo35->Fill(Vml(4)/sqrt(Cml(4,4))); 

    if (debug_) {   //--------------------------------- debug print ----------
      
      std::cout<<" diag 0[ "<<C0(0,0)<<" "<<C0(1,1)<<" "<<C0(2,2)<<" "<<C0(3,3)<<" "
	       <<C0(4,4)<<" "<<C0(5,5)<<" ]"<<std::endl;
      std::cout<<" diag e[ "<<Ce(0,0)<<" "<<Ce(1,1)<<" "<<Ce(2,2)<<" "<<Ce(3,3)<<" "
	       <<Ce(4,4)<<" "<<Ce(5,5)<<" ]"<<std::endl;
      std::cout<<" diag 1[ "<<C1(0,0)<<" "<<C1(1,1)<<" "<<C1(2,2)<<" "<<C1(3,3)<<" "
	       <<C1(4,4)<<" "<<C1(5,5)<<" ]"<<std::endl;
      std::cout<<" Rm   "<<Rm.x()<<" "<<Rm.y()<<" "<<Rm.z()
	       <<" Pm   "<<Pm.x()<<" "<<Pm.y()<<" "<<Pm.z()<<std::endl;
      std::cout<<" Rt   "<<Rt.x()<<" "<<Rt.y()<<" "<<Rt.z()
	       <<" Pt   "<<Pt.x()<<" "<<Pt.y()<<" "<<Pt.z()<<std::endl;
      std::cout<<" Nl*(Rm-Rt) "<<Nl.dot(Rm-Rt)<<std::endl;
      std::cout<<" resR "<<resR.x()<<" "<<resR.y()<<" "<<resR.z()
	       <<" resP "<<resP.x()<<" "<<resP.y()<<" "<<resP.z()<<std::endl;       
      std::cout<<" Rm-t "<<(Rm-Rt).x()<<" "<<(Rm-Rt).y()<<" "<<(Rm-Rt).z()
	       <<" Pm-t "<<(Pm-Pt).x()<<" "<<(Pm-Pt).y()<<" "<<(Pm-Pt).z()<<std::endl;       
      std::cout<<" Vm   "<<Vm<<std::endl;
      std::cout<<" +-   "<<sqrt(Cm(0,0))<<" "<<sqrt(Cm(1,1))<<" "<<sqrt(Cm(2,2))<<" "
	       <<sqrt(Cm(3,3))<<" "<<sqrt(Cm(4,4))<<" "<<sqrt(Cm(5,5))<<std::endl;
      std::cout<<" Pmuon Ptrack dP/Ptrack "<<itMuon->outerTrack()->p()<<" "
	       <<itMuon->track()->outerP()<<" "
	       <<(itMuon->outerTrack()->p() - 
		  itMuon->track()->outerP())/itMuon->track()->outerP()<<std::endl;
      std::cout<<" cov matrix "<<std::endl;
      std::cout<<Cm<<std::endl;
      std::cout<<" diag [ "<<Cm(0,0)<<" "<<Cm(1,1)<<" "<<Cm(2,2)<<" "<<Cm(3,3)<<" "
	       <<Cm(4,4)<<" "<<Cm(5,5)<<" ]"<<std::endl;
      
      AlgebraicSymMatrix66 Ro;
      double Diag[6];
      for(int i=0; i<=5; i++) Diag[i] = sqrt(Cm(i,i));
      for(int i=0; i<=5; i++)
	for(int j=0; j<=5; j++)
	  Ro(i,j) = Cm(i,j)/Diag[i]/Diag[j];
      std::cout<<" correlation matrix "<<std::endl;
      std::cout<<Ro<<std::endl;
      
      AlgebraicSymMatrix66 CmI;
      for(int i=0; i<=5; i++)
	for(int j=0; j<=5; j++)
	  CmI(i,j) = Cm(i,j);
      
      bool ierr = !CmI.Invert();
      if( ierr ) { if(alarm || debug_)
	  std::cout<<" Error inverse covariance matrix !!!!!!!!!!!"<<std::endl;
	continue;
      }       
      std::cout<<" inverse cov matrix "<<std::endl;
      std::cout<<Cm<<std::endl;
      
      double chi = ROOT::Math::Similarity(Vm, CmI);
      std::cout<<" chi chi_d "<<chi<<" "<<chi_d<<std::endl;
    }  // end of debug_ printout  --------------------------------------------
    
  } // end loop on selected muons, i.e. Jim's globalMuon

} //end of analyzeTrackTrack


// ------- method to analyze recoTrack & trajectoryMuon of Global Muon ------
void GlobalTrackerMuonAlignment::analyzeTrackTrajectory
(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;  
  //debug_ = true;
  bool info = false;
  bool alarm = false;
  //bool alarm = true;
  double PI = 3.1415927;
 
  //-*-*-*-*-*-*-
  cosmicMuonMode_ = true; // true: both Cosmic and IsolatedMuon are treated with 1,2 tracks
  //cosmicMuonMode_ = false; // true: both Cosmic and IsolatedMuon are treated with 1,2 tracks
  //-*-*-*-*-*-*-
  isolatedMuonMode_ = !cosmicMuonMode_; //true: only IsolatedMuon are treated with 1 track
 
  N_event++;
  if (info || debug_) {
    std::cout << "----- event " << N_event << " -- tracks "<<N_track<<" ---";
    if (cosmicMuonMode_) std::cout << " treated as CosmicMu ";
    if (isolatedMuonMode_) std::cout <<" treated as IsolatedMu ";
    std::cout << std::endl;
  }

  Handle<reco::TrackCollection> tracks;
  Handle<reco::TrackCollection> muons;
  Handle<reco::TrackCollection> gmuons;
  Handle<reco::MuonCollection> smuons;

  if (collectionIsolated){
    iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "TrackerOnly",  tracks);
    iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "StandAlone",  muons);
    iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "GlobalMuon",  gmuons);
    iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "SelectedMuons",  smuons);
  }
  else if (collectionCosmic){
    iEvent.getByLabel("ALCARECOMuAlGlobalCosmics", "TrackerOnly",  tracks);
    iEvent.getByLabel("ALCARECOMuAlGlobalCosmics", "StandAlone",  muons);
    iEvent.getByLabel("ALCARECOMuAlGlobalCosmics", "GlobalMuon",  gmuons);
    iEvent.getByLabel("ALCARECOMuAlGlobalCosmics", "SelectedMuons",  smuons);
  }
  else{
    iEvent.getByLabel(trackTags_,tracks);
    iEvent.getByLabel(muonTags_,muons);
    iEvent.getByLabel(gmuonTags_,gmuons);
    iEvent.getByLabel(smuonTags_,smuons);
  }  

  if (debug_) {
    std::cout << " ievBunch " << iEvent.bunchCrossing()
	      << " runN " << (int)iEvent.run() <<std::endl;
    std::cout << " N tracks s/amu gmu selmu "<<tracks->size() << " " <<muons->size()
              << " "<<gmuons->size() << " " << smuons->size() << std::endl;
    for (MuonCollection::const_iterator itMuon = smuons->begin();
	itMuon != smuons->end();
	 ++itMuon) {
      std::cout << " is isolatValid Matches " << itMuon->isIsolationValid()
		<< " " <<itMuon->isMatchesValid() << std::endl;
    }
  }

  if (isolatedMuonMode_) {                // ---- Only 1 Isolated Muon --------
    if (tracks->size() != 1) return;  
    if (muons->size()  != 1) return;  
    if (gmuons->size() != 1) return; 
    if (smuons->size() != 1) return; 
  }
  
  if (cosmicMuonMode_){                // ---- 1,2 Cosmic Muon --------
    if (smuons->size() > 2) return; 
    if (tracks->size() != smuons->size()) return;  
    if (muons->size() != smuons->size()) return;  
    if (gmuons->size() != smuons->size()) return;  
  }
  
  // ok mc_isolated_mu
  //edm::ESHandle<TrackerGeometry> trackerGeometry;
  //iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);
  // ok mc_isolated_mu
  //edm::ESHandle<DTGeometry> dtGeometry;
  //iSetup.get<MuonGeometryRecord>().get(dtGeometry);
  // don't work
  //edm::ESHandle<CSCGeometry> cscGeometry;
  //iSetup.get<MuonGeometryRecord>().get(cscGeometry);
  
  if (watchTrackingGeometry_.check(iSetup) || !trackingGeometry_) {
    edm::ESHandle<GlobalTrackingGeometry> trackingGeometry;
    iSetup.get<GlobalTrackingGeometryRecord>().get(trackingGeometry);
    trackingGeometry_ = &*trackingGeometry;
    theTrackingGeometry = trackingGeometry;
  }
  
  if (watchMagneticFieldRecord_.check(iSetup) || !magneticField_) {
    edm::ESHandle<MagneticField> magneticField;
    iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
    magneticField_ = &*magneticField;
  }

  if (watchGlobalPositionRcd_.check(iSetup) || !globalPositionRcd_) {
    edm::ESHandle<Alignments> globalPositionRcd;
    iSetup.get<GlobalPositionRcd>().get(globalPositionRcd);
    globalPositionRcd_ = &*globalPositionRcd;
    for (std::vector<AlignTransform>::const_iterator i 
	   = globalPositionRcd_->m_align.begin();
	 i != globalPositionRcd_->m_align.end();  ++i){
      if(DetId(DetId::Tracker).rawId() == i->rawId()) iteratorTrackerRcd = i;
      if(DetId(DetId::Muon).rawId() == i->rawId()) iteratorMuonRcd = i;
      if(DetId(DetId::Ecal).rawId() == i->rawId()) iteratorEcalRcd = i;
      if(DetId(DetId::Hcal).rawId() == i->rawId()) iteratorHcalRcd = i;
    }
    if(debug_){
      std::cout << "=== iteratorTrackerRcd " << iteratorTrackerRcd->rawId()<<" ====\n" 
		<< " translation " << iteratorTrackerRcd->translation()<<"\n" 
		<< " angles " << iteratorTrackerRcd->rotation().eulerAngles() << std::endl;
      std::cout << iteratorTrackerRcd->rotation() << std::endl;
      std::cout << "=== iteratorMuonRcd " << iteratorMuonRcd->rawId()<<" ====\n" 
		<< " translation " << iteratorMuonRcd->translation()<<"\n" 
		<< " angles " << iteratorMuonRcd->rotation().eulerAngles() << std::endl;
      std::cout << iteratorMuonRcd->rotation() << std::endl;
    }   
  } // end of GlobalPositionRcd
  
  ESHandle<Propagator> propagator;
  iSetup.get<TrackingComponentsRecord>().get(propagator_, propagator);

  SteppingHelixPropagator alongStHePr = 
    SteppingHelixPropagator(magneticField_, alongMomentum);  
  SteppingHelixPropagator oppositeStHePr = 
    SteppingHelixPropagator(magneticField_, oppositeToMomentum);  

  //double tolerance = 5.e-5;
  defaultRKPropagator::Product  aprod( magneticField_, alongMomentum, 5.e-5); auto & alongRKPr = aprod.propagator;
  defaultRKPropagator::Product  oprod( magneticField_, oppositeToMomentum, 5.e-5); auto & oppositeRKPr = oprod.propagator;

  float epsilon = 5.;
  SmartPropagator alongSmPr = 
    SmartPropagator(alongRKPr, alongStHePr, magneticField_, alongMomentum, epsilon);
  SmartPropagator oppositeSmPr = 
    SmartPropagator(oppositeRKPr, oppositeStHePr, magneticField_, oppositeToMomentum, epsilon);

  if(defineFitter){
    if(debug_)
      std::cout<<" ............... DEFINE FITTER ..................."<<std::endl;
    KFUpdator* theUpdator = new KFUpdator();
    //Chi2MeasurementEstimator* theEstimator = new Chi2MeasurementEstimator(30);
    Chi2MeasurementEstimator* theEstimator = new Chi2MeasurementEstimator(100000,100000);
    theFitter = new KFTrajectoryFitter(alongSmPr, 
				       *theUpdator, 
				       *theEstimator);
    theSmoother = new KFTrajectorySmoother(alongSmPr,
					   *theUpdator,     
					   *theEstimator);
    theFitterOp = new KFTrajectoryFitter(oppositeSmPr, 
					 *theUpdator, 
					 *theEstimator);
    theSmootherOp = new KFTrajectorySmoother(oppositeSmPr,
					     *theUpdator,     
					     *theEstimator);
    
    edm::ESHandle<TransientTrackingRecHitBuilder> builder;
    iSetup.get<TransientRecHitRecord>().get("WithTrackAngle",builder);
    this->TTRHBuilder = &(*builder);
    this->MuRHBuilder = new MuonTransientTrackingRecHitBuilder(theTrackingGeometry);
    if(debug_){
      std::cout<<" theTrackingGeometry.isValid() "<<theTrackingGeometry.isValid()<<std::endl;
      std::cout<< "get also the MuonTransientTrackingRecHitBuilder" << "\n";
      std::cout<< "get also the TransientTrackingRecHitBuilder" << "\n";
    }
    defineFitter = false;
  }

  // ................................................ selected/global muon
  //itMuon -->  Jim's globalMuon  
  for(MuonCollection::const_iterator itMuon = smuons->begin();
      itMuon != smuons->end(); ++itMuon) {
    
    if(debug_){
      std::cout<<" mu gM is GM Mu SaM tM "<<itMuon->isGlobalMuon()<<" "
	       <<itMuon->isMuon()<<" "<<itMuon->isStandAloneMuon()
	       <<" "<<itMuon->isTrackerMuon()<<" "
	       <<std::endl;
    }
    
    // get information about the innermost muon hit -------------------------
    TransientTrack muTT(itMuon->outerTrack(), magneticField_, trackingGeometry_);
    TrajectoryStateOnSurface innerMuTSOS = muTT.innermostMeasurementState();
    TrajectoryStateOnSurface outerMuTSOS = muTT.outermostMeasurementState();
    
    // get information about the outermost tracker hit -----------------------
    TransientTrack trackTT(itMuon->track(), magneticField_, trackingGeometry_);
    TrajectoryStateOnSurface outerTrackTSOS = trackTT.outermostMeasurementState();
    TrajectoryStateOnSurface innerTrackTSOS = trackTT.innermostMeasurementState();
    
    GlobalPoint pointTrackIn  = innerTrackTSOS.globalPosition();
    GlobalPoint pointTrackOut = outerTrackTSOS.globalPosition();
    float lenghtTrack = (pointTrackOut-pointTrackIn).mag();
    GlobalPoint pointMuonIn  = innerMuTSOS.globalPosition();
    GlobalPoint pointMuonOut = outerMuTSOS.globalPosition();
    float lenghtMuon = (pointMuonOut - pointMuonIn).mag();
    GlobalVector momentumTrackOut = outerTrackTSOS.globalMomentum();
    GlobalVector momentumTrackIn = innerTrackTSOS.globalMomentum();
    float distanceInIn   = (pointTrackIn  - pointMuonIn).mag();
    float distanceInOut  = (pointTrackIn  - pointMuonOut).mag();
    float distanceOutIn  = (pointTrackOut - pointMuonIn).mag();
    float distanceOutOut = (pointTrackOut - pointMuonOut).mag();

    if(debug_){
      std::cout<<" pointTrackIn "<<pointTrackIn<<std::endl;
      std::cout<<"          Out "<<pointTrackOut<<" lenght "<<lenghtTrack<<std::endl;
      std::cout<<" point MuonIn "<<pointMuonIn<<std::endl;
      std::cout<<"          Out "<<pointMuonOut<<" lenght "<<lenghtMuon<<std::endl;
      std::cout<<" momeTrackIn Out "<<momentumTrackIn<<" "<<momentumTrackOut<<std::endl;
      std::cout<<" doi io ii oo "<<distanceOutIn<<" "<<distanceInOut<<" "
	       <<distanceInIn<<" "<<distanceOutOut<<std::endl; 
  }   

    if(lenghtTrack < 90.) continue;
    if(lenghtMuon < 300.) continue;
    if(innerMuTSOS.globalMomentum().mag() < 5. || outerMuTSOS.globalMomentum().mag() < 5.) continue;
    if(momentumTrackIn.mag() < 15. || momentumTrackOut.mag() < 15.) continue;
    if(trackTT.charge() != muTT.charge()) continue;

    if(debug_)
      std::cout<<" passed lenght momentum cuts"<<std::endl;

    GlobalVector GRm, GPm, Nl, Rm, Pm, Rt, Pt, Rt0;
    AlgebraicSymMatrix66 Cm, C0, Ce, C1;

    //extrapolationT = propagator->propagate(outerTrackTSOS, refSurface);
    TrajectoryStateOnSurface extrapolationT;
    int tsosMuonIf = 0;

    TrajectoryStateOnSurface muonFittedTSOS;
    TrajectoryStateOnSurface trackFittedTSOS;

    if( isolatedMuonMode_ ){      //------------------------------- Isolated Muon --- Out-In --
      if(debug_) std::cout<<" ------ Isolated (out-in) ----- "<<std::endl;
      const Surface& refSurface = innerMuTSOS.surface(); 
      ConstReferenceCountingPointer<TangentPlane> 
	tpMuLocal(refSurface.tangentPlane(innerMuTSOS.localPosition()));
      Nl = tpMuLocal->normalVector();
      ConstReferenceCountingPointer<TangentPlane> 
      tpMuGlobal(refSurface.tangentPlane(innerMuTSOS.globalPosition()));
      GlobalVector Ng = tpMuGlobal->normalVector();
      Surface* surf = (Surface*)&refSurface;
      const Plane* refPlane = dynamic_cast<Plane*>(surf); 
      GlobalVector Nlp = refPlane->normalVector();
      
      if(debug_){
	std::cout<<" Nlocal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
	 <<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
	std::cout<<"  global "<<Ng.x()<<" "<<Ng.y()<<" "<<Ng.z()<<std::endl;
	std::cout<<"      lp "<<Nlp.x()<<" "<<Nlp.y()<<" "<<Nlp.z()<<std::endl;
	//std::cout<<" Nlocal Nglobal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
	// <<Ng.x()<<" "<<Ng.y()<<" "<<Ng.z()
	//<<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
      }

      //                          extrapolation to innermost muon hit     

      //extrapolationT = alongSmPr.propagate(outerTrackTSOS, refSurface);
      if(!refitTrack_)
	extrapolationT = alongSmPr.propagate(outerTrackTSOS, refSurface);
      else{
	GlobalTrackerMuonAlignment::trackFitter(itMuon->track(),
						trackTT,alongMomentum,trackFittedTSOS);
	if(trackFittedTSOS.isValid())
	  extrapolationT = alongSmPr.propagate(trackFittedTSOS, refSurface);	   
      }   
      
      if(!extrapolationT.isValid()){
	if(false & alarm)
	  std::cout<<" !!!!!!!!! Catastrophe Out-In extrapolationT.isValid "
	    //<<"\a\a\a\a\a\a\a\a"<<extrapolationT.isValid()
		   <<std::endl;
	continue;
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
      Cm = AlgebraicSymMatrix66 (extrapolationT.cartesianError().matrix() + 
				 innerMuTSOS.cartesianError().matrix());
      C0 = AlgebraicSymMatrix66(outerTrackTSOS.cartesianError().matrix()); 		   
      Ce = AlgebraicSymMatrix66(extrapolationT.cartesianError().matrix()); 
      C1 = AlgebraicSymMatrix66(innerMuTSOS.cartesianError().matrix());

      if(refitMuon_)
	GlobalTrackerMuonAlignment::muonFitter(itMuon->outerTrack(),muTT,oppositeToMomentum,muonFittedTSOS);

    }  //                    -------------------------------   end Isolated Muon -- Out - In  ---
    
    
    if( cosmicMuonMode_ ){      //------------------------------- Cosmic Muon -----
      
      if((distanceOutIn <= distanceInOut) & (distanceOutIn <= distanceInIn) & 
	 (distanceOutIn <= distanceOutOut)){             // ----- Out - In ------    
	if(debug_) std::cout<<" -----    Out - In -----"<<std::endl;

        const Surface& refSurface = innerMuTSOS.surface(); 
	ConstReferenceCountingPointer<TangentPlane> 
	  tpMuGlobal(refSurface.tangentPlane(innerMuTSOS.globalPosition()));
	Nl = tpMuGlobal->normalVector();

	//                          extrapolation to innermost muon hit     
	//extrapolationT = alongSmPr.propagate(outerTrackTSOS, refSurface);
	if(!refitTrack_)
	  extrapolationT = alongSmPr.propagate(outerTrackTSOS, refSurface);
	else{
	  GlobalTrackerMuonAlignment::trackFitter(itMuon->track(),
						  trackTT,alongMomentum,trackFittedTSOS);
	  if(trackFittedTSOS.isValid())
	    extrapolationT = alongSmPr.propagate(trackFittedTSOS, refSurface);	   
	}  

	if(!extrapolationT.isValid()){
	  if(false & alarm)
	    std::cout<<" !!!!!!!!! Catastrophe Out-In extrapolationT.isValid "
	      //<<"\a\a\a\a\a\a\a\a"<<extrapolationT.isValid()
		     <<std::endl;
	  continue;
	}
	if(debug_)
	  std::cout<<" extrapolationT.isValid "<<extrapolationT.isValid()<<std::endl; 

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
	Cm = AlgebraicSymMatrix66 (extrapolationT.cartesianError().matrix() + 
				   innerMuTSOS.cartesianError().matrix());
	C0 = AlgebraicSymMatrix66(outerTrackTSOS.cartesianError().matrix()); 		   
	Ce = AlgebraicSymMatrix66(extrapolationT.cartesianError().matrix()); 
	C1 = AlgebraicSymMatrix66(innerMuTSOS.cartesianError().matrix());
	
	if(refitMuon_)
	  GlobalTrackerMuonAlignment::muonFitter(itMuon->outerTrack(),muTT,oppositeToMomentum,muonFittedTSOS);

	if(false & debug_){
	  //std::cout<<" ->propDir "<<propagator->propagationDirection()<<std::endl;
	  PropagationDirectionChooser Chooser;
	  std::cout<<" propDirCh "
		   <<Chooser.operator()(*outerTrackTSOS.freeState(), refSurface)
		   <<" Ch == along "<<(alongMomentum == 
				       Chooser.operator()(*outerTrackTSOS.freeState(), refSurface))
		   <<std::endl;
	  std::cout<<" --- Nlocal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
		   <<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
	}
      }           //                                       enf of ---- Out - In -----
      

      else if((distanceInOut <= distanceInIn) & (distanceInOut <= distanceOutIn) & 
	      (distanceInOut <= distanceOutOut)){            // ----- In - Out ------    
	if(debug_) std::cout<<" -----    In - Out -----"<<std::endl;

	const Surface& refSurface = outerMuTSOS.surface(); 
	ConstReferenceCountingPointer<TangentPlane> 
	  tpMuGlobal(refSurface.tangentPlane(outerMuTSOS.globalPosition()));
	Nl = tpMuGlobal->normalVector();
		
	//                          extrapolation to outermost muon hit     
	//extrapolationT = oppositeSmPr.propagate(innerTrackTSOS, refSurface);
	if(!refitTrack_)
	  extrapolationT = oppositeSmPr.propagate(innerTrackTSOS, refSurface);
	else{
	  GlobalTrackerMuonAlignment::trackFitter(itMuon->track(),
						trackTT,oppositeToMomentum,trackFittedTSOS);
	if(trackFittedTSOS.isValid())
	  extrapolationT = oppositeSmPr.propagate(trackFittedTSOS, refSurface);	   
      }   

	if(!extrapolationT.isValid()){
	  if(false & alarm)
	    std::cout<<" !!!!!!!!! Catastrophe Out-In extrapolationT.isValid "
	      <<"\a\a\a\a\a\a\a\a"<<extrapolationT.isValid()
		     <<std::endl;
	  continue;
	}
	if(debug_)
	  std::cout<<" extrapolationT.isValid "<<extrapolationT.isValid()<<std::endl; 
	
	tsosMuonIf = 2;
	Rt = GlobalVector((extrapolationT.globalPosition()).x(),
			  (extrapolationT.globalPosition()).y(),
			  (extrapolationT.globalPosition()).z());
	
	Pt = extrapolationT.globalMomentum();
	//                          global parameters of muon
	GRm = GlobalVector((outerMuTSOS.globalPosition()).x(),
			   (outerMuTSOS.globalPosition()).y(),
			   (outerMuTSOS.globalPosition()).z());
	GPm = outerMuTSOS.globalMomentum();
	Rt0 = GlobalVector((innerTrackTSOS.globalPosition()).x(),
			   (innerTrackTSOS.globalPosition()).y(),
			   (innerTrackTSOS.globalPosition()).z());
	Cm = AlgebraicSymMatrix66 (extrapolationT.cartesianError().matrix() + 
				   outerMuTSOS.cartesianError().matrix());
	C0 = AlgebraicSymMatrix66(innerTrackTSOS.cartesianError().matrix()); 		   
	Ce = AlgebraicSymMatrix66(extrapolationT.cartesianError().matrix()); 
	C1 = AlgebraicSymMatrix66(outerMuTSOS.cartesianError().matrix());
	
	if(refitMuon_)
	  GlobalTrackerMuonAlignment::muonFitter(itMuon->outerTrack(),muTT,alongMomentum,muonFittedTSOS);

	if(false & debug_){
	  //std::cout<<" ->propDir "<<propagator->propagationDirection()<<std::endl;
	  PropagationDirectionChooser Chooser;
	  std::cout<<" propDirCh "
		   <<Chooser.operator()(*innerTrackTSOS.freeState(), refSurface)
		   <<" Ch == oppisite "<<(oppositeToMomentum == 
					  Chooser.operator()(*innerTrackTSOS.freeState(), refSurface))
		   <<std::endl;
	  std::cout<<" --- Nlocal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
		   <<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
	}
      }           //                                        enf of ---- In - Out -----

      else if((distanceOutOut <= distanceInOut) & (distanceOutOut <= distanceInIn) & 
	      (distanceOutOut <= distanceOutIn)){          // ----- Out - Out ------    
	if(debug_) std::cout<<" -----    Out - Out -----"<<std::endl;

	// reject: momentum of track has opposite direction to muon track
	continue;

	const Surface& refSurface = outerMuTSOS.surface(); 
	ConstReferenceCountingPointer<TangentPlane> 
	  tpMuGlobal(refSurface.tangentPlane(outerMuTSOS.globalPosition()));
	Nl = tpMuGlobal->normalVector();
		
	//                          extrapolation to outermost muon hit     
	extrapolationT = alongSmPr.propagate(outerTrackTSOS, refSurface);

	if(!extrapolationT.isValid()){
	  if(alarm)
	    std::cout<<" !!!!!!!!! Catastrophe Out-Out extrapolationT.isValid "
		     <<"\a\a\a\a\a\a\a\a"<<extrapolationT.isValid()
		     <<std::endl;
	  continue;
	}
	if(debug_)
	  std::cout<<" extrapolationT.isValid "<<extrapolationT.isValid()<<std::endl; 

	tsosMuonIf = 2;
	Rt = GlobalVector((extrapolationT.globalPosition()).x(),
			  (extrapolationT.globalPosition()).y(),
			  (extrapolationT.globalPosition()).z());
	
	Pt = extrapolationT.globalMomentum();
	//                          global parameters of muon
	GRm = GlobalVector((outerMuTSOS.globalPosition()).x(),
			   (outerMuTSOS.globalPosition()).y(),
			   (outerMuTSOS.globalPosition()).z());
	GPm = outerMuTSOS.globalMomentum();
	Rt0 = GlobalVector((outerTrackTSOS.globalPosition()).x(),
			   (outerTrackTSOS.globalPosition()).y(),
			   (outerTrackTSOS.globalPosition()).z());
	Cm = AlgebraicSymMatrix66 (extrapolationT.cartesianError().matrix() + 
				   outerMuTSOS.cartesianError().matrix());
	C0 = AlgebraicSymMatrix66(outerTrackTSOS.cartesianError().matrix()); 		   
	Ce = AlgebraicSymMatrix66(extrapolationT.cartesianError().matrix()); 
	C1 = AlgebraicSymMatrix66(outerMuTSOS.cartesianError().matrix());
	
	if(debug_){
	  PropagationDirectionChooser Chooser;
	  std::cout<<" propDirCh "
		   <<Chooser.operator()(*outerTrackTSOS.freeState(), refSurface)
		   <<" Ch == along "<<(alongMomentum == 
				       Chooser.operator()(*outerTrackTSOS.freeState(), refSurface))
		   <<std::endl;
	  std::cout<<" --- Nlocal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
		   <<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
	  std::cout<<"     Nornal "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "
		   <<" alfa "<<int(asin(Nl.x())*57.29578)<<std::endl;
	}
      }           //                                       enf of ---- Out - Out -----
      else{
	if(alarm)
	  std::cout<<" ------- !!!!!!!!!!!!!!! No proper Out - In \a\a\a\a\a\a\a"<<std::endl; 
	continue;
      }
      
    }  //                     -------------------------------   end Cosmic Muon -----
    
    if(tsosMuonIf == 0) {if(info) {std::cout<<"No tsosMuon !!!!!!"<<std::endl; continue;}}
    TrajectoryStateOnSurface tsosMuon;
    if(tsosMuonIf == 1) tsosMuon = muTT.innermostMeasurementState();
    else tsosMuon = muTT.outermostMeasurementState();

    //GlobalTrackerMuonAlignment::misalignMuon(GRm, GPm, Nl, Rt, Rm, Pm);
    AlgebraicVector4 LPRm;  // muon local (dx/dz, dy/dz, x, y)
    GlobalTrackerMuonAlignment::misalignMuonL
    (GRm, GPm, Nl, Rt, Rm, Pm, LPRm, extrapolationT, tsosMuon);

    if(refitTrack_){
      if(!trackFittedTSOS.isValid()){ 
	if(info) std::cout<<" =================  trackFittedTSOS notValid !!!!!!!! "<<std::endl;
	continue;}
      if(debug_) this->debugTrajectorySOS(" trackFittedTSOS ", trackFittedTSOS);
    }

    if(refitMuon_){
      if(!muonFittedTSOS.isValid()){ 
	if(info) std::cout<<" =================  muonFittedTSOS notValid !!!!!!!! "<<std::endl;
	continue;}
      if(debug_) this->debugTrajectorySOS(" muonFittedTSOS ", muonFittedTSOS);
      Rm = GlobalVector((muonFittedTSOS.globalPosition()).x(),
			(muonFittedTSOS.globalPosition()).y(),
			(muonFittedTSOS.globalPosition()).z());
      Pm = muonFittedTSOS.globalMomentum();
      LPRm = AlgebraicVector4(muonFittedTSOS.localParameters().vector()(1),
			      muonFittedTSOS.localParameters().vector()(2),
			      muonFittedTSOS.localParameters().vector()(3),
			      muonFittedTSOS.localParameters().vector()(4)); 
    }
    GlobalVector resR = Rm - Rt;
    GlobalVector resP0 = Pm - Pt;
    GlobalVector resP = Pm / Pm.mag() - Pt / Pt.mag();
    float RelMomResidual = (Pm.mag() - Pt.mag()) / (Pt.mag() + 1.e-6);;

    AlgebraicVector6 Vm;
    Vm(0) = resR.x();  Vm(1) = resR.y();  Vm(2) = resR.z(); 
    Vm(3) = resP0.x();  Vm(4) = resP0.y();  Vm(5) = resP0.z(); 
    float Rmuon = Rm.perp();
    float Zmuon = Rm.z();
    float alfa_x = atan2(Nl.x(),Nl.y())*57.29578;
    
    if(debug_){
      std::cout<<" Nx Ny Nz alfa_x "<<Nl.x()<<" "<<Nl.y()<<" "<<Nl.z()<<" "<<alfa_x<<std::endl;
      //std::cout<<" Rm "<<Rm<<std::endl<<" Rt "<<Rt<<std::endl<<" resR "<<resR<<std::endl
      //       <<" resP "<<resP<<" dp/p "<<RelMomResidual<<std::endl;
    }

    double chi_d = 0;
    for(int i=0; i<=5; i++) chi_d += Vm(i)*Vm(i)/Cm(i,i);
        
    AlgebraicVector5 Vml(tsosMuon.localParameters().vector() 
		       - extrapolationT.localParameters().vector());
    AlgebraicSymMatrix55 m(tsosMuon.localError().matrix() + extrapolationT.localError().matrix());
    AlgebraicSymMatrix55 Cml(tsosMuon.localError().matrix() + extrapolationT.localError().matrix());
    bool ierrLoc = !m.Invert();
    if (ierrLoc && debug_ && info) { 
      std::cout<< " ==== Error inverting Local covariance matrix ==== "<<std::cout;
      continue;}
    double chi_Loc = ROOT::Math::Similarity(Vml,m);
    if(debug_)
      std::cout<<" chi_Loc px/pz/err "<<chi_Loc<<" "<<Vml(1)/sqrt(Cml(1,1))<<std::endl;

    if(Pt.mag() < 15.) continue;
    if(Pm.mag() <  5.) continue;
 
    //if(Pt.mag() < 30.) continue;          // momenum cut  < 30GeV
    //if(Pt.mag() < 60.) continue;          // momenum cut  < 30GeV
    //if(Pt.mag() > 50.) continue;          //  momenum cut > 50GeV
    //if(Pt.mag() > 100.) continue;          //  momenum cut > 100GeV
    //if(trackTT.charge() < 0) continue;    // select positive charge
    //if(trackTT.charge() > 0) continue;    // select negative charge
    
    //if(fabs(resR.x()) > 5.) continue;     // strong cut  X 
    //if(fabs(resR.y()) > 5.) continue;     //             Y 
    //if(fabs(resR.z()) > 5.) continue;     //             Z
    //if(fabs(resR.mag()) > 7.5) continue;   //            dR

    //if(fabs(RelMomResidual) > 0.5) continue;
    if(fabs(resR.x()) > 20.) continue;
    if(fabs(resR.y()) > 20.) continue;
    if(fabs(resR.z()) > 20.) continue;
    if(fabs(resR.mag()) > 30.) continue;
    if(fabs(resP.x()) > 0.06) continue;
    if(fabs(resP.y()) > 0.06) continue;
    if(fabs(resP.z()) > 0.06) continue;
    if(chi_d > 40.) continue;

    //                                            select Barrel 
    //if(Rmuon < 400. || Rmuon > 450.) continue; 
    //if(Zmuon < -600. || Zmuon > 600.) continue;
    //if(fabs(Nl.z()) > 0.95) continue;  
    //MuSelect = " Barrel";
    //                                                  EndCap1
    //if(Rmuon < 120. || Rmuon > 450.) continue;
    //if(Zmuon < -720.) continue;
    //if(Zmuon > -580.) continue;
    //if(fabs(Nl.z()) < 0.95) continue;  
    //MuSelect = " EndCap1";
    //                                                  EndCap2
    //if(Rmuon < 120. || Rmuon > 450.) continue;
    //if(Zmuon >  720.) continue;
    //if(Zmuon <  580.) continue;
    //if(fabs(Nl.z()) < 0.95) continue;  
    //MuSelect = " EndCap2";
    //                                                 select All
    if(Rmuon < 120. || Rmuon > 450.) continue;  
    if(Zmuon < -720. || Zmuon > 720.) continue;
    MuSelect = " Barrel+EndCaps";

    if(debug_)
      std::cout<<" .............. passed all cuts"<<std::endl;
        
    N_track++;
    //                     gradient and Hessian for each track
    
    GlobalTrackerMuonAlignment::gradientGlobalAlg(Rt, Pt, Rm, Nl, Cm);
    GlobalTrackerMuonAlignment::gradientGlobal(Rt, Pt, Rm, Pm, Nl, Cm); 

    CLHEP::HepSymMatrix covLoc(4,0);
    for(int i=1; i<=4; i++)
      for(int j=1; j<=i; j++){
	covLoc(i,j) = (tsosMuon.localError().matrix()
		       + extrapolationT.localError().matrix())(i,j);
	//if(i != j) Cov(i,j) = 0.;
      }

    const Surface& refSurface = tsosMuon.surface(); 
    CLHEP::HepMatrix rotLoc (3,3,0);
    rotLoc(1,1) = refSurface.rotation().xx();
    rotLoc(1,2) = refSurface.rotation().xy();
    rotLoc(1,3) = refSurface.rotation().xz();
    
    rotLoc(2,1) = refSurface.rotation().yx();
    rotLoc(2,2) = refSurface.rotation().yy();
    rotLoc(2,3) = refSurface.rotation().yz();
    
    rotLoc(3,1) = refSurface.rotation().zx();
    rotLoc(3,2) = refSurface.rotation().zy();
    rotLoc(3,3) = refSurface.rotation().zz();
    
    CLHEP::HepVector posLoc(3);
    posLoc(1) = refSurface.position().x();
    posLoc(2) = refSurface.position().y();
    posLoc(3) = refSurface.position().z();

    GlobalTrackerMuonAlignment::gradientLocal(Rt, Pt, Rm, Pm, Nl, covLoc, rotLoc, posLoc, LPRm);

    if(debug_){
      std::cout<<" Norm   "<<Nl<<std::endl;
      std::cout<<" posLoc "<<posLoc.T()<<std::endl;
      std::cout<<" rotLoc "<<rotLoc<<std::endl;
    }

    // -----------------------------------------------------  fill histogram 
    histo->Fill(itMuon->track()->pt()); 
    
    //histo2->Fill(itMuon->track()->outerP()); 
    histo2->Fill(Pt.mag()); 
    histo3->Fill((PI/2.-itMuon->track()->outerTheta())); 
    histo4->Fill(itMuon->track()->phi()); 
    histo5->Fill(Rmuon); 
    histo6->Fill(Zmuon); 
    histo7->Fill(RelMomResidual); 
    //histo8->Fill(chi); 
    histo8->Fill(chi_d); 
    
    histo101->Fill(Zmuon, Rmuon); 
    histo101->Fill(Rt0.z(), Rt0.perp()); 
    histo102->Fill(Rt0.x(), Rt0.y()); 
    histo102->Fill(Rm.x(), Rm.y()); 
    
    histo11->Fill(resR.mag()); 
    if(fabs(Nl.x()) < 0.98) histo12->Fill(resR.x()); 
    if(fabs(Nl.y()) < 0.98) histo13->Fill(resR.y()); 
    if(fabs(Nl.z()) < 0.98) histo14->Fill(resR.z()); 
    histo15->Fill(resP.x()); 
    histo16->Fill(resP.y()); 
    histo17->Fill(resP.z()); 
    
    if((fabs(Nl.x()) < 0.98) && (fabs(Nl.y()) < 0.98) &&(fabs(Nl.z()) < 0.98))
      { 
	histo18->Fill(sqrt(C0(0,0))); 
	histo19->Fill(sqrt(C1(0,0))); 
	histo20->Fill(sqrt(C1(0,0)+Ce(0,0))); 		   
      }
    if(fabs(Nl.x()) < 0.98) histo21->Fill(Vm(0)/sqrt(Cm(0,0))); 
    if(fabs(Nl.y()) < 0.98) histo22->Fill(Vm(1)/sqrt(Cm(1,1))); 
    if(fabs(Nl.z()) < 0.98) histo23->Fill(Vm(2)/sqrt(Cm(2,2))); 
    histo24->Fill(Vm(3)/sqrt(C1(3,3)+Ce(3,3))); 
    histo25->Fill(Vm(4)/sqrt(C1(4,4)+Ce(4,4))); 
    histo26->Fill(Vm(5)/sqrt(C1(5,5)+Ce(5,5))); 
    histo27->Fill(Nl.x()); 
    histo28->Fill(Nl.y()); 
    histo29->Fill(lenghtTrack); 
    histo30->Fill(lenghtMuon);     
    histo31->Fill(chi_Loc);
    histo32->Fill(Vml(1)/sqrt(Cml(1,1))); 
    histo33->Fill(Vml(2)/sqrt(Cml(2,2))); 
    histo34->Fill(Vml(3)/sqrt(Cml(3,3))); 
    histo35->Fill(Vml(4)/sqrt(Cml(4,4))); 

    if (debug_) {   //--------------------------------- debug print ----------
      
      std::cout<<" diag 0[ "<<C0(0,0)<<" "<<C0(1,1)<<" "<<C0(2,2)<<" "<<C0(3,3)<<" "
	       <<C0(4,4)<<" "<<C0(5,5)<<" ]"<<std::endl;
      std::cout<<" diag e[ "<<Ce(0,0)<<" "<<Ce(1,1)<<" "<<Ce(2,2)<<" "<<Ce(3,3)<<" "
	       <<Ce(4,4)<<" "<<Ce(5,5)<<" ]"<<std::endl;
      std::cout<<" diag 1[ "<<C1(0,0)<<" "<<C1(1,1)<<" "<<C1(2,2)<<" "<<C1(3,3)<<" "
	       <<C1(4,4)<<" "<<C1(5,5)<<" ]"<<std::endl;
      std::cout<<" Rm   "<<Rm.x()<<" "<<Rm.y()<<" "<<Rm.z()
	       <<" Pm   "<<Pm.x()<<" "<<Pm.y()<<" "<<Pm.z()<<std::endl;
      std::cout<<" Rt   "<<Rt.x()<<" "<<Rt.y()<<" "<<Rt.z()
	       <<" Pt   "<<Pt.x()<<" "<<Pt.y()<<" "<<Pt.z()<<std::endl;
      std::cout<<" Nl*(Rm-Rt) "<<Nl.dot(Rm-Rt)<<std::endl;
      std::cout<<" resR "<<resR.x()<<" "<<resR.y()<<" "<<resR.z()
	       <<" resP "<<resP.x()<<" "<<resP.y()<<" "<<resP.z()<<std::endl;       
      std::cout<<" Rm-t "<<(Rm-Rt).x()<<" "<<(Rm-Rt).y()<<" "<<(Rm-Rt).z()
	       <<" Pm-t "<<(Pm-Pt).x()<<" "<<(Pm-Pt).y()<<" "<<(Pm-Pt).z()<<std::endl;       
      std::cout<<" Vm   "<<Vm<<std::endl;
      std::cout<<" +-   "<<sqrt(Cm(0,0))<<" "<<sqrt(Cm(1,1))<<" "<<sqrt(Cm(2,2))<<" "
	       <<sqrt(Cm(3,3))<<" "<<sqrt(Cm(4,4))<<" "<<sqrt(Cm(5,5))<<std::endl;
      std::cout<<" Pmuon Ptrack dP/Ptrack "<<itMuon->outerTrack()->p()<<" "
	       <<itMuon->track()->outerP()<<" "
	       <<(itMuon->outerTrack()->p() - 
		  itMuon->track()->outerP())/itMuon->track()->outerP()<<std::endl;
      std::cout<<" cov matrix "<<std::endl;
      std::cout<<Cm<<std::endl;
      std::cout<<" diag [ "<<Cm(0,0)<<" "<<Cm(1,1)<<" "<<Cm(2,2)<<" "<<Cm(3,3)<<" "
	       <<Cm(4,4)<<" "<<Cm(5,5)<<" ]"<<std::endl;
      
      AlgebraicSymMatrix66 Ro;
      double Diag[6];
      for(int i=0; i<=5; i++) Diag[i] = sqrt(Cm(i,i));
      for(int i=0; i<=5; i++)
	for(int j=0; j<=5; j++)
	  Ro(i,j) = Cm(i,j)/Diag[i]/Diag[j];
      std::cout<<" correlation matrix "<<std::endl;
      std::cout<<Ro<<std::endl;
      
      AlgebraicSymMatrix66 CmI;
      for(int i=0; i<=5; i++)
	for(int j=0; j<=5; j++)
	  CmI(i,j) = Cm(i,j);
      
      bool ierr = !CmI.Invert();
      if( ierr ) { if(alarm || debug_)
	  std::cout<<" Error inverse covariance matrix !!!!!!!!!!!"<<std::endl;
	continue;
      }       
      std::cout<<" inverse cov matrix "<<std::endl;
      std::cout<<Cm<<std::endl;
      
      double chi = ROOT::Math::Similarity(Vm, CmI);
      std::cout<<" chi chi_d "<<chi<<" "<<chi_d<<std::endl;
    }  // end of debug_ printout  --------------------------------------------
    
  } // end loop on selected muons, i.e. Jim's globalMuon

} //end of analyzeTrackTrajectory


// ----  calculate gradient and Hessian matrix (algebraic) to search global shifts ------
void 
GlobalTrackerMuonAlignment::gradientGlobalAlg(GlobalVector& Rt, GlobalVector& Pt, 
				     GlobalVector& Rm, GlobalVector& Nl, 
				     AlgebraicSymMatrix66& Cm)
{
  // ---------------------------- Calculate Information matrix and Gfree vector
  // Information == Hessian , Gfree == gradient of the objective function
  
  AlgebraicMatrix33 Jac; 
  AlgebraicVector3 Wi, R_m, R_t, P_t, Norm, dR;   
  
  R_m(0)  = Rm.x();  R_m(1)  = Rm.y();  R_m(2)  = Rm.z(); 
  R_t(0)  = Rt.x();  R_t(1)  = Rt.y();  R_t(2)  = Rt.z(); 
  P_t(0)  = Pt.x();  P_t(1)  = Pt.y();  P_t(2)  = Pt.z(); 
  Norm(0) = Nl.x();  Norm(1) = Nl.y();  Norm(2) = Nl.z(); 
  
  for(int i=0; i<=2; i++){
    if(Cm(i,i) > 1.e-20)
      Wi(i) = 1./Cm(i,i);
    else Wi(i) = 1.e-10;
    dR(i) = R_m(i) - R_t(i);
  }
  
  float PtN = P_t(0)*Norm(0) + P_t(1)*Norm(1) + P_t(2)*Norm(2);
  
  Jac(0,0) = 1. - P_t(0)*Norm(0)/PtN;
  Jac(0,1) =    - P_t(0)*Norm(1)/PtN;
  Jac(0,2) =    - P_t(0)*Norm(2)/PtN;
  
  Jac(1,0) =    - P_t(1)*Norm(0)/PtN;
  Jac(1,1) = 1. - P_t(1)*Norm(1)/PtN;
  Jac(1,2) =    - P_t(1)*Norm(2)/PtN;
  
  Jac(2,0) =    - P_t(2)*Norm(0)/PtN;
  Jac(2,1) =    - P_t(2)*Norm(1)/PtN;
  Jac(2,2) = 1. - P_t(2)*Norm(2)/PtN;
  
  AlgebraicSymMatrix33 Itr;
  
  for(int i=0; i<=2; i++) 
    for(int j=0; j<=2; j++){
      if(j < i) continue;
      Itr(i,j) = 0.;
      //std::cout<<" ij "<<i<<" "<<j<<std::endl;	 
      for(int k=0; k<=2; k++){
	Itr(i,j) += Jac(k,i)*Wi(k)*Jac(k,j);}}
  
  for(int i=0; i<=2; i++) 
    for(int j=0; j<=2; j++){
      if(j < i) continue;
      Inf(i,j) += Itr(i,j);}
  
  AlgebraicVector3 Gtr(0., 0., 0.);
  for(int i=0; i<=2; i++)
    for(int k=0; k<=2; k++) Gtr(i) += dR(k)*Wi(k)*Jac(k,i);
  for(int i=0; i<=2; i++) Gfr(i) += Gtr(i); 

  if(debug_){
   std::cout<<" Wi  "<<Wi<<std::endl;     
   std::cout<<" N   "<<Norm<<std::endl;
   std::cout<<" P_t "<<P_t<<std::endl;
   std::cout<<" (Pt*N) "<<PtN<<std::endl;  
   std::cout<<" dR  "<<dR<<std::endl;
   std::cout<<" +/- "<<1./sqrt(Wi(0))<<" "<<1./sqrt(Wi(1))<<" "<<1./sqrt(Wi(2))
         <<" "<<std::endl;
   std::cout<<" Jacobian dr/ddx "<<std::endl;
   std::cout<<Jac<<std::endl;
   std::cout<<" G-- "<<Gtr<<std::endl;
   std::cout<<" Itrack "<<std::endl;
   std::cout<<Itr<<std::endl;
   std::cout<<" Gfr "<<Gfr<<std::endl;
   std::cout<<" -- Inf --"<<std::endl;
   std::cout<<Inf<<std::endl;
  }

  return;
}

// ----  calculate gradient and Hessian matrix in global parameters ------
void 
GlobalTrackerMuonAlignment::gradientGlobal(GlobalVector& GRt, GlobalVector& GPt, 
				  GlobalVector& GRm, GlobalVector& GPm, 
				  GlobalVector& GNorm, AlgebraicSymMatrix66 & GCov)
{
  // we search for 6D global correction vector (d, a), where
  //                        3D vector of shihts d 
  //               3D vector of rotation angles    a   

  //bool alarm = true;
  bool alarm = false;
  bool info = false;

  int Nd = 6;     // dimension of vector of alignment pararmeters, d 
  
  //double PtMom = GPt.mag();
  CLHEP::HepSymMatrix w(Nd,0);
  for (int i=1; i <= Nd; i++)
    for (int j=1; j <= Nd; j++){
      if(j <= i ) w(i,j) = GCov(i-1, j-1);
      //if(i >= 3) w(i,j) /= PtMom;
      //if(j >= 3) w(i,j) /= PtMom;
      if((i == j) && (i<=3) && (GCov(i-1, j-1) < 1.e-20))  w(i,j) = 1.e20; // w=0  
      if(i != j) w(i,j) = 0.;                // use diaginal elements    
    }

  //GPt /= GPt.mag();
  //GPm /= GPm.mag();   // end of transform

  CLHEP::HepVector V(Nd), Rt(3), Pt(3), Rm(3), Pm(3), Norm(3);
  Rt(1) = GRt.x(); Rt(2) = GRt.y();  Rt(3) = GRt.z();
  Pt(1) = GPt.x(); Pt(2) = GPt.y();  Pt(3) = GPt.z();
  Rm(1) = GRm.x(); Rm(2) = GRm.y();  Rm(3) = GRm.z();
  Pm(1) = GPm.x(); Pm(2) = GPm.y();  Pm(3) = GPm.z();
   Norm(1) = GNorm.x(); Norm(2) = GNorm.y(); Norm(3) = GNorm.z(); 

  V = dsum(Rm - Rt, Pm - Pt) ;   //std::cout<<" V "<<V.T()<<std::endl;
  
  //double PmN = CLHEP::dot(Pm, Norm);
  double PmN = CLHEP_dot(Pm, Norm);

  CLHEP::HepMatrix Jac(Nd,Nd,0);
  for (int i=1; i <= 3; i++)
    for (int j=1; j <= 3; j++){
      Jac(i,j) =  Pm(i)*Norm(j) / PmN;
      if(i == j ) Jac(i,j) -= 1.;
    } 

  //                                            dp/da                   
  Jac(4,4) =     0.;   // dpx/dax
  Jac(5,4) = -Pm(3);   // dpy/dax
  Jac(6,4) =  Pm(2);   // dpz/dax
  Jac(4,5) =  Pm(3);   // dpx/day
  Jac(5,5) =     0.;   // dpy/day
  Jac(6,5) = -Pm(1);   // dpz/day
  Jac(4,6) = -Pm(2);   // dpx/daz
  Jac(5,6) =  Pm(1);   // dpy/daz
  Jac(6,6) =     0.;   // dpz/daz

  CLHEP::HepVector dsda(3);
  dsda(1) = (Norm(2)*Rm(3) - Norm(3)*Rm(2)) / PmN;
  dsda(2) = (Norm(3)*Rm(1) - Norm(1)*Rm(3)) / PmN;
  dsda(3) = (Norm(1)*Rm(2) - Norm(2)*Rm(1)) / PmN;

  //                                             dr/da  
  Jac(1,4) =          Pm(1)*dsda(1);  // drx/dax
  Jac(2,4) = -Rm(3) + Pm(2)*dsda(1);  // dry/dax
  Jac(3,4) =  Rm(2) + Pm(3)*dsda(1);  // drz/dax

  Jac(1,5) =  Rm(3) + Pm(1)*dsda(2);  // drx/day
  Jac(2,5) =          Pm(2)*dsda(2);  // dry/day
  Jac(3,5) = -Rm(1) + Pm(3)*dsda(2);  // drz/day

  Jac(1,6) = -Rm(2) + Pm(1)*dsda(3);  // drx/daz
  Jac(2,6) =  Rm(1) + Pm(2)*dsda(3);  // dry/daz
  Jac(3,6) =          Pm(3)*dsda(3);  // drz/daz

  CLHEP::HepSymMatrix W(Nd,0);
  int ierr;
  W = w.inverse(ierr);
  if(ierr != 0) { if(alarm)
      std::cout<<" gradientGlobal: inversion of matrix w fail "<<std::endl;
    return;
  }

  CLHEP::HepMatrix W_Jac(Nd,Nd,0);
  W_Jac = Jac.T() * W;
  
  CLHEP::HepVector grad3(Nd);
  grad3 = W_Jac * V;

  CLHEP::HepMatrix hess3(Nd,Nd);
  hess3 = Jac.T() * W * Jac;
  //hess3(4,4) = 1.e-10;  hess3(5,5) = 1.e-10;  hess3(6,6) = 1.e-10; //????????????????

  Grad += grad3;
  Hess += hess3;

  CLHEP::HepVector d3I = CLHEP::solve(Hess, -Grad);
  int iEr3I;
  CLHEP::HepMatrix Errd3I = Hess.inverse(iEr3I);
  if( iEr3I != 0) { if(alarm)
      std::cout<<" gradientGlobal error inverse  Hess matrix !!!!!!!!!!!"<<std::endl;
  }

  if(info || debug_){
    std::cout<<" dG   "<<d3I(1)<<" "<<d3I(2)<<" "<<d3I(3)<<" "<<d3I(4)<<" "
	     <<d3I(5)<<" "<<d3I(6)<<std::endl;;
    std::cout<<" +-   "<<sqrt(Errd3I(1,1))<<" "<<sqrt(Errd3I(2,2))<<" "<<sqrt(Errd3I(3,3))
	     <<" "<<sqrt(Errd3I(4,4))<<" "<<sqrt(Errd3I(5,5))<<" "<<sqrt(Errd3I(6,6))
	     <<std::endl;
  }

#ifdef CHECK_OF_DERIVATIVES
  //              --------------------    check of derivatives
  
  CLHEP::HepVector d(3,0), a(3,0); 
  CLHEP::HepMatrix T(3,3,1);

  CLHEP::HepMatrix Ti = T.T();
  //double A = CLHEP::dot(Ti*Pm, Norm);
  //double B = CLHEP::dot((Rt -Ti*Rm + Ti*d), Norm);
  double A = CLHEP_dot(Ti*Pm, Norm);
  double B = CLHEP_dot((Rt -Ti*Rm + Ti*d), Norm);
  double s0 = B / A;

  CLHEP::HepVector r0(3,0), p0(3,0);
  r0 = Ti*Rm - Ti*d + s0*(Ti*Pm) - Rt;
  p0 = Ti*Pm - Pt;

  double delta = 0.0001;

  int ii = 3; d(ii) += delta; // d
  //T(2,3) += delta;   T(3,2) -= delta;  int ii = 1; // a1
  //T(3,1) += delta;   T(1,3) -= delta;   int ii = 2; // a2
  //T(1,2) += delta;   T(2,1) -= delta;   int ii = 3; // a2
  Ti = T.T();
  //A = CLHEP::dot(Ti*Pm, Norm);
  //B = CLHEP::dot((Rt -Ti*Rm + Ti*d), Norm);
  A = CLHEP_dot(Ti*Pm, Norm);
  B = CLHEP_dot((Rt -Ti*Rm + Ti*d), Norm);
  double s = B / A;

  CLHEP::HepVector r(3,0), p(3,0);
  r = Ti*Rm - Ti*d + s*(Ti*Pm) - Rt;
  p = Ti*Pm - Pt;

  std::cout<<" s0 s num dsda("<<ii<<") "<<s0<<" "<<s<<" "
  	   <<(s-s0)/delta<<" "<<dsda(ii)<<endl;
  // d(r,p) / d shift
  std::cout<<" -- An d(r,p)/d("<<ii<<") "<<Jac(1,ii)<<" "<<Jac(2,ii)<<" "
     <<Jac(3,ii)<<" "<<Jac(4,ii)<<" "<<Jac(5,ii)<<" "
     <<Jac(6,ii)<<std::endl;
  std::cout<<"    Nu d(r,p)/d("<<ii<<") "<<(r(1)-r0(1))/delta<<" "
     <<(r(2)-r0(2))/delta<<" "<<(r(3)-r0(3))/delta<<" "
     <<(p(1)-p0(1))/delta<<" "<<(p(2)-p0(2))/delta<<" "
     <<(p(3)-p0(3))/delta<<std::endl;
  // d(r,p) / d angle
  std::cout<<" -- An d(r,p)/a("<<ii<<") "<<Jac(1,ii+3)<<" "<<Jac(2,ii+3)<<" "
	   <<Jac(3,ii+3)<<" "<<Jac(4,ii+3)<<" "<<Jac(5,ii+3)<<" "
	   <<Jac(6,ii+3)<<std::endl;
  std::cout<<"    Nu d(r,p)/a("<<ii<<") "<<(r(1)-r0(1))/delta<<" "
	   <<(r(2)-r0(2))/delta<<" "<<(r(3)-r0(3))/delta<<" "
	   <<(p(1)-p0(1))/delta<<" "<<(p(2)-p0(2))/delta<<" "
	   <<(p(3)-p0(3))/delta<<std::endl;
  //               -----------------------------  end of check
#endif

  return;
} // end gradientGlobal


// ----  calculate gradient and Hessian matrix in local parameters ------
void 
GlobalTrackerMuonAlignment::gradientLocal(GlobalVector& GRt, GlobalVector& GPt, 
					  GlobalVector& GRm, GlobalVector& GPm, 
					  GlobalVector& GNorm, 
					  CLHEP::HepSymMatrix& covLoc,
					  CLHEP::HepMatrix& rotLoc,
					  CLHEP::HepVector& R0,
					  AlgebraicVector4& LPRm) 
{
  // we search for 6D global correction vector (d, a), where
  //                        3D vector of shihts d 
  //               3D vector of rotation angles    a   

  bool alarm = true;
  //bool alarm = false;
  bool info = false;

  if(debug_)
    std::cout<<" gradientLocal "<<std::endl; 

  /*
  const Surface& refSurface = tsosMuon.surface();

  CLHEP::HepMatrix rotLoc (3,3,0);
  rotLoc(1,1) = refSurface.rotation().xx();
  rotLoc(1,2) = refSurface.rotation().xy();
  rotLoc(1,3) = refSurface.rotation().xz();
  
  rotLoc(2,1) = refSurface.rotation().yx();
  rotLoc(2,2) = refSurface.rotation().yy();
  rotLoc(2,3) = refSurface.rotation().yz();
  
  rotLoc(3,1) = refSurface.rotation().zx();
  rotLoc(3,2) = refSurface.rotation().zy();
  rotLoc(3,3) = refSurface.rotation().zz();
  */  

  CLHEP::HepVector Rt(3), Pt(3), Rm(3), Pm(3), Norm(3);
  Rt(1) = GRt.x(); Rt(2) = GRt.y();  Rt(3) = GRt.z();
  Pt(1) = GPt.x(); Pt(2) = GPt.y();  Pt(3) = GPt.z();
  Rm(1) = GRm.x(); Rm(2) = GRm.y();  Rm(3) = GRm.z();
  Pm(1) = GPm.x(); Pm(2) = GPm.y();  Pm(3) = GPm.z();
  Norm(1) = GNorm.x(); Norm(2) = GNorm.y(); Norm(3) = GNorm.z(); 

  CLHEP::HepVector V(4), Rml(3), Pml(3), Rtl(3), Ptl(3);

  /*
  R0(1) = refSurface.position().x();
  R0(2) = refSurface.position().y();
  R0(3) = refSurface.position().z();
  */ 

  Rml = rotLoc * (Rm - R0);
  Rtl = rotLoc * (Rt - R0);
  Pml = rotLoc * Pm;
  Ptl = rotLoc * Pt;
  
  V(1) = LPRm(0) - Ptl(1)/Ptl(3); 
  V(2) = LPRm(1) - Ptl(2)/Ptl(3);   
  V(3) = LPRm(2) - Rtl(1);   
  V(4) = LPRm(3) - Rtl(2); 

  /*
  CLHEP::HepSymMatrix Cov(4,0), W(4,0);
  for(int i=1; i<=4; i++)
    for(int j=1; j<=i; j++){
      Cov(i,j) = (tsosMuon.localError().matrix()
		  + tsosTrack.localError().matrix())(i,j);
      //if(i != j) Cov(i,j) = 0.;
      //if((i == j) && ((i==1) || (i==2))) Cov(i,j) = 100.;
      //if((i == j) && ((i==3) || (i==4))) Cov(i,j) = 10000.;
    }
  W = Cov;
  */

  CLHEP::HepSymMatrix W = covLoc;

  int ierr;
  W.invert(ierr);
  if(ierr != 0) { if(alarm)
      std::cout<<" gradientLocal: inversion of matrix W fail "<<std::endl;
    return;
  }
  
  //                               JacobianCartesianToLocal
 
  //AlgebraicMatrix56 jacobian   // differ from calculation above 
  //= JacobianCartesianToLocal::JacobianCartesianToLocal 
  //(refSurface, tsosTrack.localParameters()).jacobian();  
  //for(int i=1; i<=4; i++) for(int j=1; j<=6; j++){
  //int j1 = j - 1; JacToLoc(i,j) = jacobian(i, j1);}
  
  CLHEP::HepMatrix JacToLoc(4,6,0);
  for(int i=1; i<=2; i++)
    for(int j=1; j<=3; j++){
      JacToLoc(i,j+3) = (rotLoc(i,j) - rotLoc(3,j)*Pml(i)/Pml(3))/Pml(3); 
      JacToLoc(i+2,j) = rotLoc(i,j);
    }

  //                                    JacobianCorrectionsToCartesian
  //double PmN = CLHEP::dot(Pm, Norm);
  double PmN = CLHEP_dot(Pm, Norm);

  CLHEP::HepMatrix Jac(6,6,0);
  for (int i=1; i <= 3; i++)
    for (int j=1; j <= 3; j++){
      Jac(i,j) =  Pm(i)*Norm(j) / PmN;
      if(i == j ) Jac(i,j) -= 1.;
    }  

  //                                            dp/da                   
  Jac(4,4) =     0.;   // dpx/dax
  Jac(5,4) = -Pm(3);   // dpy/dax
  Jac(6,4) =  Pm(2);   // dpz/dax
  Jac(4,5) =  Pm(3);   // dpx/day
  Jac(5,5) =     0.;   // dpy/day
  Jac(6,5) = -Pm(1);   // dpz/day
  Jac(4,6) = -Pm(2);   // dpx/daz
  Jac(5,6) =  Pm(1);   // dpy/daz
  Jac(6,6) =     0.;   // dpz/daz

  CLHEP::HepVector dsda(3);
  dsda(1) = (Norm(2)*Rm(3) - Norm(3)*Rm(2)) / PmN;
  dsda(2) = (Norm(3)*Rm(1) - Norm(1)*Rm(3)) / PmN;
  dsda(3) = (Norm(1)*Rm(2) - Norm(2)*Rm(1)) / PmN;

  //                                             dr/da  
  Jac(1,4) =          Pm(1)*dsda(1);  // drx/dax
  Jac(2,4) = -Rm(3) + Pm(2)*dsda(1);  // dry/dax
  Jac(3,4) =  Rm(2) + Pm(3)*dsda(1);  // drz/dax

  Jac(1,5) =  Rm(3) + Pm(1)*dsda(2);  // drx/day
  Jac(2,5) =          Pm(2)*dsda(2);  // dry/day
  Jac(3,5) = -Rm(1) + Pm(3)*dsda(2);  // drz/day

  Jac(1,6) = -Rm(2) + Pm(1)*dsda(3);  // drx/daz
  Jac(2,6) =  Rm(1) + Pm(2)*dsda(3);  // dry/daz
  Jac(3,6) =          Pm(3)*dsda(3);  // drz/daz

  //                                   JacobianCorrectionToLocal
  CLHEP::HepMatrix JacCorLoc(4,6,0);
  JacCorLoc = JacToLoc * Jac;

  //                                   gradient and Hessian 
  CLHEP::HepMatrix W_Jac(6,4,0);
  W_Jac = JacCorLoc.T() * W;
  
  CLHEP::HepVector gradL(6);
  gradL = W_Jac * V;

  CLHEP::HepMatrix hessL(6,6);
  hessL = JacCorLoc.T() * W * JacCorLoc;

  GradL += gradL;
  HessL += hessL;

  CLHEP::HepVector dLI = CLHEP::solve(HessL, -GradL);
  int iErI;
  CLHEP::HepMatrix ErrdLI = HessL.inverse(iErI);
  if( iErI != 0) { if(alarm)
      std::cout<<" gradLocal Error inverse  Hess matrix !!!!!!!!!!!"<<std::endl;
  }

  if(info || debug_){
    std::cout<<" dL   "<<dLI(1)<<" "<<dLI(2)<<" "<<dLI(3)<<" "<<dLI(4)<<" "
	     <<dLI(5)<<" "<<dLI(6)<<std::endl;;
    std::cout<<" +-   "<<sqrt(ErrdLI(1,1))<<" "<<sqrt(ErrdLI(2,2))<<" "<<sqrt(ErrdLI(3,3))
	     <<" "<<sqrt(ErrdLI(4,4))<<" "<<sqrt(ErrdLI(5,5))<<" "<<sqrt(ErrdLI(6,6))
	     <<std::endl;
  }
 
  if(debug_){
    //std::cout<<" dV(da3) {"<<-JacCorLoc(1,6)*0.002<<" "<<-JacCorLoc(2,6)*0.002<<" "
    //   <<-JacCorLoc(3,6)*0.002<<" "<<-JacCorLoc(4,6)*0.002<<"}"<<std::endl;
    //std::cout<<" JacCLo {"<<JacCorLoc(1,6)<<" "<<JacCorLoc(2,6)<<" "
    //   <<JacCorLoc(3,6)<<" "<<JacCorLoc(4,6)<<"}"<<std::endl;
    //std::cout<<"Jpx/yx  {"<<Jac(4,6)/Pm(3)<<" "<<Jac(5,6)/Pm(3)<<" "
    //   <<Jac(1,6)<<" "<<Jac(2,6)<<"}"<<std::endl;
    //std::cout<<"Jac(,a3){"<<Jac(1,6)<<" "<<Jac(2,6)<<" "<<Jac(3,6)<<" "<<Jac(4,6)<<" "
    //   <<Jac(5,6)<<" "<<Jac(6,6)<<std::endl;
    int i=5;  
    if(GNorm.z() > 0.95)
      std::cout<<" Ecap1   N "<<GNorm<<std::endl; 
    else if(GNorm.z() < -0.95)
      std::cout<<" Ecap2   N "<<GNorm<<std::endl; 
    else
      std::cout<<" Barrel  N "<<GNorm<<std::endl; 
    std::cout<<" JacCLo(i,"<<i<<") = {"<<JacCorLoc(1,i)<<" "<<JacCorLoc(2,i)<<" "
     <<JacCorLoc(3,i)<<" "<<JacCorLoc(4,i)<<"}"<<std::endl;
    std::cout<<"   rotLoc "<<rotLoc<<std::endl;
    std::cout<<" position "<<R0<<std::endl;
    std::cout<<" Pm,l "<<Pm.T()<<" "<<Pml(1)/Pml(3)<<" "<<Pml(2)/Pml(3)<<std::endl;
    std::cout<<" Pt,l "<<Pt.T()<<" "<<Ptl(1)/Ptl(3)<<" "<<Ptl(2)/Ptl(3)<<std::endl;
    std::cout<<" V  "<<V.T()<<std::endl;  
    std::cout<<" Cov \n"<<covLoc<<std::endl;
    std::cout<<" W*Cov "<<W*covLoc<<std::endl;
    //std::cout<<" JacCarToLoc = drldrc \n"<<
    //JacobianCartesianToLocal::JacobianCartesianToLocal
    //(refSurface, tsosTrack.localParameters()).jacobian()<<std::endl;
    std::cout<<" JacToLoc "<<JacToLoc<<std::endl;
    
  }

#ifdef CHECK_OF_JACOBIAN_CARTESIAN_TO_LOCAL
  //---------------------- check of derivatives
  CLHEP::HepVector V0(4,0);
  V0(1) = Pml(1)/Pml(3) - Ptl(1)/Ptl(3); 
  V0(2) = Pml(2)/Pml(3) - Ptl(2)/Ptl(3);   
  V0(3) = Rml(1) - Rtl(1);   
  V0(4) = Rml(2) - Rtl(2); 
  int ii = 3; float delta = 0.01;
  CLHEP::HepVector V1(4,0);
  if(ii <= 3) {Rm(ii) += delta; Rml = rotLoc * (Rm - R0);}
  else {Pm(ii-3) += delta; Pml = rotLoc * Pm;}
  //if(ii <= 3) {Rt(ii) += delta; Rtl = rotLoc * (Rt - R0);}
  //else {Pt(ii-3) += delta; Ptl = rotLoc * Pt;}
  V1(1) = Pml(1)/Pml(3) - Ptl(1)/Ptl(3); 
  V1(2) = Pml(2)/Pml(3) - Ptl(2)/Ptl(3);   
  V1(3) = Rml(1) - Rtl(1);   
  V1(4) = Rml(2) - Rtl(2); 

  if(GNorm.z() > 0.95)
    std::cout<<" Ecap1   N "<<GNorm<<std::endl; 
  else if(GNorm.z() < -0.95)
    std::cout<<" Ecap2   N "<<GNorm<<std::endl; 
  else
    std::cout<<" Barrel  N "<<GNorm<<std::endl; 
  std::cout<<" dldc Num (i,"<<ii<<") "<<(V1(1)-V0(1))/delta<<" "<<(V1(2)-V0(2))/delta<<" "
	   <<(V1(3)-V0(3))/delta<<" "<<(V1(4)-V0(4))/delta<<std::endl;
  std::cout<<" dldc Ana (i,"<<ii<<") "<<JacToLoc(1,ii)<<" "<<JacToLoc(2,ii)<<" "
	   <<JacToLoc(3,ii)<<" "<<JacToLoc(4,ii)<<std::endl;
  //float dtxdpx = (rotLoc(1,1) - rotLoc(3,1)*Pml(1)/Pml(3))/Pml(3);
  //float dtydpx = (rotLoc(2,1) - rotLoc(3,2)*Pml(2)/Pml(3))/Pml(3);
  //std::cout<<" dtx/dpx dty/ "<<dtxdpx<<" "<<dtydpx<<std::endl;
#endif						  

  return;
} // end gradientLocal


// ----  simulate a misalignment of muon system   ------
void 
GlobalTrackerMuonAlignment::misalignMuon(GlobalVector& GRm, GlobalVector& GPm, GlobalVector& Nl, 
				GlobalVector& Rt, GlobalVector& Rm, GlobalVector& Pm)
{
  CLHEP::HepVector d(3,0), a(3,0), Norm(3), Rm0(3), Pm0(3), Rm1(3), Pm1(3), 
    Rmi(3), Pmi(3) ; 
  
  d(1)=0.0; d(2)=0.0; d(3)=0.0; a(1)=0.0000; a(2)=0.0000; a(3)=0.0000; // zero     
  //d(1)=0.0; d(2)=0.0; d(3)=0.0; a(1)=0.0020; a(2)=0.0000; a(3)=0.0000; // a1=.002
  //d(1)=0.0; d(2)=0.0; d(3)=0.0; a(1)=0.0000; a(2)=0.0020; a(3)=0.0000; // a2=.002
  //d(1)=0.0; d(2)=0.0; d(3)=0.0; a(1)=0.0000; a(2)=0.0000; a(3)=0.0020; // a3=.002    
  //d(1)=1.0; d(2)=2.0; d(3)=3.0; a(1)=0.0000; a(2)=0.0000; a(3)=0.0000; // 12300     
  //d(1)=0.0; d(2)=0.0; d(3)=0.0; a(1)=0.0000; a(2)=0.0000; a(3)=0.0020; // a3=0.002    
  //d(1)=10.; d(2)=20.; d(3)=30.; a(1)=0.0100; a(2)=0.0200; a(3)=0.0300; // huge      
  //d(1)=10.; d(2)=20.; d(3)=30.; a(1)=0.2000; a(2)=0.2500; a(3)=0.3000; // huge angles      
  //d(1)=0.3; d(2)=0.4; d(3)=0.5; a(1)=0.0005; a(2)=0.0006; a(3)=0.0007; // 0345 0567
  //d(1)=0.3; d(2)=0.4; d(3)=0.5; a(1)=0.0001; a(2)=0.0002; a(3)=0.0003; // 0345 0123
  //d(1)=0.2; d(2)=0.2; d(3)=0.2; a(1)=0.0002; a(2)=0.0002; a(3)=0.0002; // 0111 0111

  MuGlShift = d; MuGlAngle = a;

  if((d(1) == 0.) & (d(2) == 0.) & (d(3) == 0.) & 
     (a(1) == 0.) & (a(2) == 0.) & (a(3) == 0.)){
    Rm = GRm;
    Pm = GPm;
    if(debug_)
      std::cout<<" ......   without misalignment "<<std::endl;
    return;
  }
  
  Rm0(1) = GRm.x(); Rm0(2) = GRm.y(); Rm0(3) = GRm.z();
  Pm0(1) = GPm.x(); Pm0(2) = GPm.y(); Pm0(3) = GPm.z();  
  Norm(1) = Nl.x(); Norm(2) = Nl.y(); Norm(3) = Nl.z();     
  
  CLHEP::HepMatrix T(3,3,1);
 
  //T(1,2) =  a(3); T(1,3) = -a(2);
  //T(2,1) = -a(3); T(2,3) =  a(1);
  //T(3,1) =  a(2); T(3,2) = -a(1);
  
  double s1 = std::sin(a(1)), c1 = std::cos(a(1));
  double s2 = std::sin(a(2)), c2 = std::cos(a(2));
  double s3 = std::sin(a(3)), c3 = std::cos(a(3));     
  T(1,1) =  c2 * c3;  
  T(1,2) =  c1 * s3 + s1 * s2 * c3;
  T(1,3) =  s1 * s3 - c1 * s2 * c3;
  T(2,1) = -c2 * s3; 
  T(2,2) =  c1 * c3 - s1 * s2 * s3;
  T(2,3) =  s1 * c3 + c1 * s2 * s3;
  T(3,1) =  s2;
  T(3,2) = -s1 * c2;
  T(3,3) =  c1 * c2;
  
  //int terr;
  //CLHEP::HepMatrix Ti = T.inverse(terr);
  CLHEP::HepMatrix Ti = T.T();
  CLHEP::HepVector di = -Ti * d;
  
  CLHEP::HepVector ex0(3,0), ey0(3,0), ez0(3,0), ex(3,0), ey(3,0), ez(3,0);
  ex0(1) = 1.; ey0(2) = 1.;  ez0(3) = 1.;
  ex = Ti*ex0; ey = Ti*ey0; ez = Ti*ez0;
  
  CLHEP::HepVector TiN = Ti * Norm; 
  //double si = CLHEP::dot((Ti*Rm0 - Rm0 + di), TiN) / CLHEP::dot(Pm0, TiN); 
  //Rm1(1) = CLHEP::dot(ex, Rm0 + si*Pm0 - di);
  //Rm1(2) = CLHEP::dot(ey, Rm0 + si*Pm0 - di);
  //Rm1(3) = CLHEP::dot(ez, Rm0 + si*Pm0 - di);
  double si = CLHEP_dot((Ti*Rm0 - Rm0 + di), TiN) / CLHEP_dot(Pm0, TiN); 
  Rm1(1) = CLHEP_dot(ex, Rm0 + si*Pm0 - di);
  Rm1(2) = CLHEP_dot(ey, Rm0 + si*Pm0 - di);
  Rm1(3) = CLHEP_dot(ez, Rm0 + si*Pm0 - di);
  Pm1 = T * Pm0;
  
  Rm = GlobalVector(Rm1(1), Rm1(2), Rm1(3));  
  //Pm = GlobalVector(CLHEP::dot(Pm0,ex), CLHEP::dot(Pm0, ey), CLHEP::dot(Pm0,ez));// =T*Pm0 
  Pm = GlobalVector(CLHEP_dot(Pm0,ex), CLHEP_dot(Pm0, ey), CLHEP_dot(Pm0,ez));// =T*Pm0 
  
  if(debug_){    //    -------------  debug tranformation 
    
    std::cout<<" ----- Pm "<<Pm<<std::endl;
    std::cout<<"    T*Pm0 "<<Pm1.T()<<std::endl;
    //std::cout<<" Ti*T*Pm0 "<<(Ti*(T*Pm0)).T()<<std::endl;
    
    //CLHEP::HepVector Rml = (Rm0 + si*Pm0 - di) + di;
    CLHEP::HepVector Rml = Rm1(1)*ex + Rm1(2)*ey + Rm1(3)*ez + di;
    
    CLHEP::HepVector Rt0(3); 
    Rt0(1)=Rt.x();  Rt0(2)=Rt.y();  Rt0(3)=Rt.z();      
    
    //double sl = CLHEP::dot(T*(Rt0 - Rml), T*Norm) / CLHEP::dot(Ti * Pm1, Norm);     
    double sl = CLHEP_dot(T*(Rt0 - Rml), T*Norm) / CLHEP_dot(Ti * Pm1, Norm);     
    CLHEP::HepVector Rl = Rml + sl*(Ti*Pm1);
    
    //double A = CLHEP::dot(Ti*Pm1, Norm);
    //double B = CLHEP::dot(Rt0, Norm) - CLHEP::dot(Ti*Rm1, Norm) 
    //+ CLHEP::dot(Ti*d, Norm); 
    double A = CLHEP_dot(Ti*Pm1, Norm);
    double B = CLHEP_dot(Rt0, Norm) - CLHEP_dot(Ti*Rm1, Norm) 
      + CLHEP_dot(Ti*d, Norm); 
    double s = B/A;
    CLHEP::HepVector Dr = Ti*Rm1 - Ti*d  + s*(Ti*Pm1) - Rm0;
    CLHEP::HepVector Dp = Ti*Pm1 - Pm0;

    CLHEP::HepVector TiR = Ti * Rm0 + di;
    CLHEP::HepVector Ri = T * TiR + d;
    
    std::cout<<" --TTi-0 "<<(Ri-Rm0).T()<<std::endl;
    std::cout<<" --  Dr  "<<Dr.T()<<endl;
    std::cout<<" --  Dp  "<<Dp.T()<<endl;
    //std::cout<<" ex "<<ex.T()<<endl;
    //std::cout<<" ey "<<ey.T()<<endl;
    //std::cout<<" ez "<<ez.T()<<endl;
    //std::cout<<" ---- T  ---- "<<T<<std::endl;
    //std::cout<<" ---- Ti ---- "<<Ti<<std::endl;
    //std::cout<<" ---- d  ---- "<<d.T()<<std::endl;
    //std::cout<<" ---- di ---- "<<di.T()<<std::endl;
    std::cout<<" -- si sl s "<<si<<" "<<sl<<" "<<s<<std::endl;
    //std::cout<<" -- si sl "<<si<<" "<<sl<<endl;
    //std::cout<<" -- si s "<<si<<" "<<s<<endl;
    std::cout<<" -- Rml-(Rm0+si*Pm0) "<<(Rml - Rm0 - si*Pm0).T()<<std::endl;
    //std::cout<<" -- Rm0 "<<Rm0.T()<<std::endl;
    //std::cout<<" -- Rm1 "<<Rm1.T()<<std::endl;
    //std::cout<<" -- Rmi "<<Rmi.T()<<std::endl;
    //std::cout<<" --siPm "<<(si*Pm0).T()<<std::endl;
    //std::cout<<" --s2Pm "<<(s2*(T * Pm1)).T()<<std::endl;
    //std::cout<<" --R1-0 "<<(Rm1-Rm0).T()<<std::endl;
    //std::cout<<" --Ri-0 "<<(Rmi-Rm0).T()<<std::endl;
    std::cout<<" --Rl-0 "<<(Rl-Rm0).T()<<std::endl;
    //std::cout<<" --Pi-0 "<<(Pmi-Pm0).T()<<std::endl;
  }      //                                --------   end of debug
  
  return;

} // end of misalignMuon

// ----  simulate a misalignment of muon system with Local  ------
void 
GlobalTrackerMuonAlignment::misalignMuonL(GlobalVector& GRm, GlobalVector& GPm, GlobalVector& Nl, 
					  GlobalVector& Rt, GlobalVector& Rm, GlobalVector& Pm,
					  AlgebraicVector4& Vm, 
					  TrajectoryStateOnSurface& tsosTrack,
					  TrajectoryStateOnSurface& tsosMuon)
{
  CLHEP::HepVector d(3,0), a(3,0), Norm(3), Rm0(3), Pm0(3), Rm1(3), Pm1(3), 
    Rmi(3), Pmi(3); 
  
  d(1)=0.0; d(2)=0.0; d(3)=0.0; a(1)=0.0000; a(2)=0.0000; a(3)=0.0000; // zero     
  //d(1)=0.0000001; d(2)=0.0; d(3)=0.0; a(1)=0.0000; a(2)=0.0000; a(3)=0.0000; // nonzero
  //d(1)=0.0; d(2)=0.0; d(3)=0.0; a(1)=0.0020; a(2)=0.0000; a(3)=0.0000; // a1=.002
  //d(1)=0.0; d(2)=0.0; d(3)=0.0; a(1)=0.0000; a(2)=0.0020; a(3)=0.0000; // a2=.002
  //d(1)=0.0; d(2)=0.0; d(3)=0.0; a(1)=0.0000; a(2)=0.0000; a(3)=0.0020; // a3=.002    
  //d(1)=0.0; d(2)=0.0; d(3)=0.0; a(1)=0.0000; a(2)=0.0000; a(3)=0.0200; // a3=.020    
  //d(1)=1.0; d(2)=2.0; d(3)=3.0; a(1)=0.0000; a(2)=0.0000; a(3)=0.0000; // 12300     
  //d(1)=0.0; d(2)=0.0; d(3)=0.0; a(1)=0.0000; a(2)=0.0000; a(3)=0.0020; // a3=0.002    
  //d(1)=10.; d(2)=20.; d(3)=30.; a(1)=0.0100; a(2)=0.0200; a(3)=0.0300; // huge      
  //d(1)=10.; d(2)=20.; d(3)=30.; a(1)=0.2000; a(2)=0.2500; a(3)=0.3000; // huge angles      
  //d(1)=0.3; d(2)=0.4; d(3)=0.5; a(1)=0.0005; a(2)=0.0006; a(3)=0.0007; // 0345 0567
  //d(1)=0.3; d(2)=0.4; d(3)=0.5; a(1)=0.0001; a(2)=0.0002; a(3)=0.0003; // 0345 0123
  //d(1)=0.1; d(2)=0.2; d(3)=0.3; a(1)=0.0003; a(2)=0.0004; a(3)=0.0005; // 0123 0345

  MuGlShift = d; MuGlAngle = a;

  if((d(1) == 0.) & (d(2) == 0.) & (d(3) == 0.) & 
     (a(1) == 0.) & (a(2) == 0.) & (a(3) == 0.)){
    Rm = GRm;
    Pm = GPm;
    AlgebraicVector4 Vm0;
    Vm = AlgebraicVector4(tsosMuon.localParameters().vector()(1),
			  tsosMuon.localParameters().vector()(2),
			  tsosMuon.localParameters().vector()(3),
			  tsosMuon.localParameters().vector()(4)); 
    if(debug_)
      std::cout<<" ......   without misalignment "<<std::endl;
    return;
  }
  
  Rm0(1) = GRm.x(); Rm0(2) = GRm.y(); Rm0(3) = GRm.z();
  Pm0(1) = GPm.x(); Pm0(2) = GPm.y(); Pm0(3) = GPm.z();  
  Norm(1) = Nl.x(); Norm(2) = Nl.y(); Norm(3) = Nl.z();     
  
  CLHEP::HepMatrix T(3,3,1);
 
  //T(1,2) =  a(3); T(1,3) = -a(2);
  //T(2,1) = -a(3); T(2,3) =  a(1);
  //T(3,1) =  a(2); T(3,2) = -a(1);
  
  double s1 = std::sin(a(1)), c1 = std::cos(a(1));
  double s2 = std::sin(a(2)), c2 = std::cos(a(2));
  double s3 = std::sin(a(3)), c3 = std::cos(a(3));     
  T(1,1) =  c2 * c3;  
  T(1,2) =  c1 * s3 + s1 * s2 * c3;
  T(1,3) =  s1 * s3 - c1 * s2 * c3;
  T(2,1) = -c2 * s3; 
  T(2,2) =  c1 * c3 - s1 * s2 * s3;
  T(2,3) =  s1 * c3 + c1 * s2 * s3;
  T(3,1) =  s2;
  T(3,2) = -s1 * c2;
  T(3,3) =  c1 * c2;

  //                    Ti, di what we have to apply for misalignment  
  //int terr;
  //CLHEP::HepMatrix Ti = T.inverse(terr);
  CLHEP::HepMatrix Ti = T.T();
  CLHEP::HepVector di = -Ti * d;
  
  CLHEP::HepVector ex0(3,0), ey0(3,0), ez0(3,0), ex(3,0), ey(3,0), ez(3,0);
  ex0(1) = 1.; ey0(2) = 1.;  ez0(3) = 1.;
  ex = Ti*ex0; ey = Ti*ey0; ez = Ti*ez0;

  CLHEP::HepVector TiN = Ti * Norm; 
  //double si = CLHEP::dot((Ti*Rm0 - Rm0 + di), TiN) / CLHEP::dot(Pm0, TiN); 
  //Rm1(1) = CLHEP::dot(ex, Rm0 + si*Pm0 - di);
  //Rm1(2) = CLHEP::dot(ey, Rm0 + si*Pm0 - di);
  //Rm1(3) = CLHEP::dot(ez, Rm0 + si*Pm0 - di);
  double si = CLHEP_dot((Ti*Rm0 - Rm0 + di), TiN) / CLHEP_dot(Pm0, TiN); 
  Rm1(1) = CLHEP_dot(ex, Rm0 + si*Pm0 - di);
  Rm1(2) = CLHEP_dot(ey, Rm0 + si*Pm0 - di);
  Rm1(3) = CLHEP_dot(ez, Rm0 + si*Pm0 - di);
  Pm1 = T * Pm0;
    
  // Global muon parameters after misalignment
  Rm = GlobalVector(Rm1(1), Rm1(2), Rm1(3));  
  //Pm = GlobalVector(CLHEP::dot(Pm0,ex), CLHEP::dot(Pm0, ey), CLHEP::dot(Pm0,ez));// =T*Pm0 
  Pm = GlobalVector(CLHEP_dot(Pm0,ex), CLHEP_dot(Pm0, ey), CLHEP_dot(Pm0,ez));// =T*Pm0 

  // Local muon parameters after misalignment
  const Surface& refSurface = tsosMuon.surface();
  CLHEP::HepMatrix Tl(3,3,0);

  Tl(1,1) = refSurface.rotation().xx();
  Tl(1,2) = refSurface.rotation().xy();
  Tl(1,3) = refSurface.rotation().xz();
  
  Tl(2,1) = refSurface.rotation().yx();
  Tl(2,2) = refSurface.rotation().yy();
  Tl(2,3) = refSurface.rotation().yz();
  
  Tl(3,1) = refSurface.rotation().zx();
  Tl(3,2) = refSurface.rotation().zy();
  Tl(3,3) = refSurface.rotation().zz();

  CLHEP::HepVector R0(3,0), newR0(3,0), newRl(3,0), newPl(3,0);
  R0(1) = refSurface.position().x();
  R0(2) = refSurface.position().y();
  R0(3) = refSurface.position().z();

  newRl = Tl * (Rm1 - R0);
  newPl = Tl * Pm1;

  Vm(0) = newPl(1)/newPl(3);
  Vm(1) = newPl(2)/newPl(3);
  Vm(2) = newRl(1);
  Vm(3) = newRl(2);
  
#ifdef CHECK_DERIVATIVES_LOCAL_VS_ANGLES
  float del = 0.0001;
  //int ii = 1; T(3,2) +=del; T(2,3) -=del;
  int ii = 2; T(3,1) -=del; T(1,3) +=del;
  //int ii = 3; T(1,2) -=del; T(2,1) +=del;
  AlgebraicVector4 Vm0 = Vm;
  CLHEP::HepVector Rm10 = Rm1, Pm10 = Pm1;;
  CLHEP::HepMatrix newTl = Tl*T;
  Ti = T.T();
  di = -Ti * d;  
  ex = Ti*ex0; ey = Ti*ey0; ez = Ti*ez0;
  TiN = Ti * Norm; 
  si = CLHEP_dot((Ti*Rm0 - Rm0 + di), TiN) / CLHEP_dot(Pm0, TiN); 
  Rm1(1) = CLHEP_dot(ex, Rm0 + si*Pm0 - di);
  Rm1(2) = CLHEP_dot(ey, Rm0 + si*Pm0 - di);
  Rm1(3) = CLHEP_dot(ez, Rm0 + si*Pm0 - di);
  Pm1 = T * Pm0;

  newRl = Tl * (Rm1 - R0);
  newPl = Tl * Pm1;

  Vm(0) = newPl(1)/newPl(3);
  Vm(1) = newPl(2)/newPl(3);
  Vm(2) = newRl(1);
  Vm(3) = newRl(2);
  std::cout<<" ========= dVm/da"<<ii<<" "<<(Vm-Vm0)*(1./del)<<std::endl;
#endif 

  if(debug_){
  //std::cout<<" dRm/da3 "<<((Rm1-Rm10)*(1./del)).T()<<" "<<((Pm1-Pm10)*(1./del)).T()<<std::endl; 
  //std::cout<<"             Norm "<<Norm.T()<<std::endl;
  std::cout<<"             ex  "<<(Tl.T()*ex0).T()<<std::endl;
  std::cout<<"             ey  "<<(Tl.T()*ey0).T()<<std::endl;
  std::cout<<"             ez  "<<(Tl.T()*ez0).T()<<std::endl;
  //std::cpot<<" 

  std::cout<<"    pxyz/p gl 0 "<<(Pm0/Pm0.norm()).T()<<std::endl;
  std::cout<<"    pxyz/p loc0 "<<(Tl*Pm0/(Tl*Pm0).norm()).T()<<std::endl;
  std::cout<<"    pxyz/p glob  "<<(Pm1/Pm1.norm()).T()<<std::endl;
  std::cout<<"    pxyz/p  loc  "<<(newPl/newPl.norm()).T()<<std::endl;

  //std::cout<<"             Rot   "<<refSurface.rotation()<<endl;
  //std::cout<<"             Tl   "<<Tl<<endl;
  std::cout<<" ---- phi g0 g1 l0 l1  "
	   <<atan2(Pm0(2),Pm0(1))<<" "<<atan2(Pm1(2),Pm1(1))<<" "
	   <<atan2((Tl*Pm0)(2),(Tl*Pm0)(1))<<" "
	   <<atan2(newPl(2),newPl(1))<<std::endl;
  std::cout<<" ---- angl Gl Loc "<<atan2(Pm1(2),Pm1(1))-atan2(Pm0(2),Pm0(1))<<" "
	   <<atan2(newPl(2),newPl(1))-atan2((Tl*Pm0)(2),(Tl*Pm0)(1))<<" "
	   <<atan2(newPl(3),newPl(2))-atan2((Tl*Pm0)(3),(Tl*Pm0)(2))<<" "
	   <<atan2(newPl(1),newPl(3))-atan2((Tl*Pm0)(1),(Tl*Pm0)(3))<<" "<<std::endl;
  }

  if(debug_){
    CLHEP::HepMatrix newTl(3,3,0);
    CLHEP::HepVector R0(3,0), newRl(3,0), newPl(3,0);
    newTl = Tl * Ti.T();
    newR0 = Ti * R0 + di;

    std::cout<<" N "<<Norm.T()<<std::endl;
    std::cout<<" dtxl yl "<<Vm(0)-tsosMuon.localParameters().vector()(1)<<" "
	     <<Vm(1)-tsosMuon.localParameters().vector()(2)<<std::endl;
    std::cout<<" dXl dYl "<<Vm(2)-tsosMuon.localParameters().vector()(3)<<" "
	     <<Vm(3)-tsosMuon.localParameters().vector()(4)<<std::endl;
    std::cout<<" localM    "<<tsosMuon.localParameters().vector()<<std::endl;  
    std::cout<<"       Vm  "<<Vm<<std::endl;
    
    
    CLHEP::HepVector Rvc(3,0), Pvc(3,0), Rvg(3,0), Pvg(3,0);
    Rvc(1) = Vm(2); Rvc(2) = Vm(3);
    Pvc(3) = tsosMuon.localParameters().pzSign() * Pm0.norm() /
      sqrt(1 + Vm(0)*Vm(0) + Vm(1)*Vm(1));
    Pvc(1) = Pvc(3) * Vm(0);
    Pvc(2) = Pvc(3) * Vm(1);
    
    Rvg = newTl.T() * Rvc + newR0;
    Pvg = newTl.T() * Pvc;
    
    std::cout<<" newPl     "<<newPl.T()<<std::endl;
    std::cout<<"    Pvc    "<<Pvc.T()<<std::endl;
    std::cout<<"    Rvg    "<<Rvg.T()<<std::endl;
    std::cout<<"    Rm1    "<<Rm1.T()<<std::endl;
    std::cout<<"    Pvg    "<<Pvg.T()<<std::endl;
    std::cout<<"    Pm1    "<<Pm1.T()<<std::endl;
  }
  
  if(debug_){    //    ----------  how to transform cartesian from shifted to correct 
    
    std::cout<<" ----- Pm "<<Pm<<std::endl;
    std::cout<<"    T*Pm0 "<<Pm1.T()<<std::endl;
    //std::cout<<" Ti*T*Pm0 "<<(Ti*(T*Pm0)).T()<<std::endl;
    
    //CLHEP::HepVector Rml = (Rm0 + si*Pm0 - di) + di;
    CLHEP::HepVector Rml = Rm1(1)*ex + Rm1(2)*ey + Rm1(3)*ez + di;
    
    CLHEP::HepVector Rt0(3); 
    Rt0(1)=Rt.x();  Rt0(2)=Rt.y();  Rt0(3)=Rt.z();      
    
    //double sl = CLHEP::dot(T*(Rt0 - Rml), T*Norm) / CLHEP::dot(Ti * Pm1, Norm);     
    double sl = CLHEP_dot(T*(Rt0 - Rml), T*Norm) / CLHEP_dot(Ti * Pm1, Norm);     
    CLHEP::HepVector Rl = Rml + sl*(Ti*Pm1);
    
    //double A = CLHEP::dot(Ti*Pm1, Norm);
    //double B = CLHEP::dot(Rt0, Norm) - CLHEP::dot(Ti*Rm1, Norm) 
    //+ CLHEP::dot(Ti*d, Norm); 
    double A = CLHEP_dot(Ti*Pm1, Norm);
    double B = CLHEP_dot(Rt0, Norm) - CLHEP_dot(Ti*Rm1, Norm) 
      + CLHEP_dot(Ti*d, Norm); 
    double s = B/A;
    CLHEP::HepVector Dr = Ti*Rm1 - Ti*d  + s*(Ti*Pm1) - Rm0;
    CLHEP::HepVector Dp = Ti*Pm1 - Pm0;

    CLHEP::HepVector TiR = Ti * Rm0 + di;
    CLHEP::HepVector Ri = T * TiR + d;
    
    std::cout<<" --TTi-0 "<<(Ri-Rm0).T()<<std::endl;
    std::cout<<" --  Dr  "<<Dr.T()<<endl;
    std::cout<<" --  Dp  "<<Dp.T()<<endl;
    //std::cout<<" ex "<<ex.T()<<endl;
    //std::cout<<" ey "<<ey.T()<<endl;
    //std::cout<<" ez "<<ez.T()<<endl;
    //std::cout<<" ---- T  ---- "<<T<<std::endl;
    //std::cout<<" ---- Ti ---- "<<Ti<<std::endl;
    //std::cout<<" ---- d  ---- "<<d.T()<<std::endl;
    //std::cout<<" ---- di ---- "<<di.T()<<std::endl;
    std::cout<<" -- si sl s "<<si<<" "<<sl<<" "<<s<<std::endl;
    //std::cout<<" -- si sl "<<si<<" "<<sl<<endl;
    //std::cout<<" -- si s "<<si<<" "<<s<<endl;
    std::cout<<" -- Rml-(Rm0+si*Pm0) "<<(Rml - Rm0 - si*Pm0).T()<<std::endl;
    //std::cout<<" -- Rm0 "<<Rm0.T()<<std::endl;
    //std::cout<<" -- Rm1 "<<Rm1.T()<<std::endl;
    //std::cout<<" -- Rmi "<<Rmi.T()<<std::endl;
    //std::cout<<" --siPm "<<(si*Pm0).T()<<std::endl;
    //std::cout<<" --s2Pm "<<(s2*(T * Pm1)).T()<<std::endl;
    //std::cout<<" --R1-0 "<<(Rm1-Rm0).T()<<std::endl;
    //std::cout<<" --Ri-0 "<<(Rmi-Rm0).T()<<std::endl;
    std::cout<<" --Rl-0 "<<(Rl-Rm0).T()<<std::endl;
    //std::cout<<" --Pi-0 "<<(Pmi-Pm0).T()<<std::endl;
  }      //                                --------   end of debug
  
  return;

}  // end misalignMuonL


// ----  refit any direction of transient track ------
void 
GlobalTrackerMuonAlignment::trackFitter(reco::TrackRef alongTr, reco::TransientTrack& alongTTr,
				       PropagationDirection direction,
				       TrajectoryStateOnSurface& trackFittedTSOS)
{
  bool info = false;
  bool smooth = false;
  
  if(info)
    std::cout<<" ......... start of trackFitter ......... "<<std::endl;  
  if(false && info){ 
    this->debugTrackHit(" Tracker track hits ", alongTr); 
    this->debugTrackHit(" Tracker TransientTrack hits ", alongTTr); 
  }

  edm::OwnVector<TrackingRecHit> recHit;
  DetId trackDetId = DetId(alongTr->innerDetId());
  for (trackingRecHit_iterator i=alongTr->recHitsBegin();i!=alongTr->recHitsEnd(); i++) {
    recHit.push_back((*i)->clone());
  }
  if(direction != alongMomentum)
    {
      edm::OwnVector<TrackingRecHit> recHitAlong = recHit;
      recHit.clear();
      for(unsigned int ihit = recHitAlong.size()-1; ihit+1>0; ihit--){
	recHit.push_back(recHitAlong[ihit]);
      }
      recHitAlong.clear();
    }
  if(info)
    std::cout<<" ... Own recHit.size() "<<recHit.size()<<" "<<trackDetId.rawId()<<std::endl;

  //MuonTransientTrackingRecHitBuilder::ConstRecHitContainer recHitTrack;
  TransientTrackingRecHit::RecHitContainer recHitTrack;
  for (trackingRecHit_iterator i=alongTr->recHitsBegin();i!=alongTr->recHitsEnd(); i++) {
    if((*i)->isValid()){
      recHitTrack.push_back(TTRHBuilder->build(&**i ));} 
    else{
      recHitTrack.push_back(TTRHBuilder->build(&**i ));}  
  }
  
  if(info)
    std::cout<<" ... recHitTrack.size() "<<recHit.size()<<" "<<trackDetId.rawId()
	     <<" ======> recHitMu "<<std::endl;

  MuonTransientTrackingRecHitBuilder::ConstRecHitContainer recHitMu = recHitTrack;
  /*
    MuonTransientTrackingRecHitBuilder::ConstRecHitContainer recHitMu =
    MuRHBuilder->build(alongTTr.recHitsBegin(), alongTTr.recHitsEnd());
  */
  if(info)
    std::cout<<" ...along.... recHitMu.size() "<<recHitMu.size()<<std::endl;
  if(direction != alongMomentum){
    MuonTransientTrackingRecHitBuilder::ConstRecHitContainer recHitMuAlong = recHitMu;
    recHitMu.clear();
    for(unsigned int ihit = recHitMuAlong.size()-1; ihit+1>0; ihit--){
      recHitMu.push_back(recHitMuAlong[ihit]);
    }
    recHitMuAlong.clear();
  }
  if(info)
    std::cout<<" ...opposite... recHitMu.size() "<<recHitMu.size()<<std::endl;

  TrajectoryStateOnSurface firstTSOS;
  if(direction == alongMomentum) firstTSOS = alongTTr.innermostMeasurementState();
  else                           firstTSOS = alongTTr.outermostMeasurementState();

  AlgebraicSymMatrix55 CovLoc;
  CovLoc(0,0) = 1.   * firstTSOS.localError().matrix()(0,0);
  CovLoc(1,1) = 10.  * firstTSOS.localError().matrix()(1,1);
  CovLoc(2,2) = 10.  * firstTSOS.localError().matrix()(2,2);
  CovLoc(3,3) = 100. * firstTSOS.localError().matrix()(3,3);
  CovLoc(4,4) = 100. * firstTSOS.localError().matrix()(4,4);
  TrajectoryStateOnSurface initialTSOS(firstTSOS.localParameters(), LocalTrajectoryError(CovLoc),
				       firstTSOS.surface(), &*magneticField_);
  
  PTrajectoryStateOnDet  PTraj = 
    trajectoryStateTransform::persistentState(initialTSOS, trackDetId.rawId());
  //const TrajectorySeed seedT(*PTraj, recHit, alongMomentum);  
  const TrajectorySeed seedT(PTraj, recHit, direction);  

  std::vector<Trajectory> trajVec;
  if(direction == alongMomentum) trajVec = theFitter->fit(seedT, recHitMu, initialTSOS);
  else                           trajVec = theFitterOp->fit(seedT, recHitMu, initialTSOS);
  
  if(info){
    this->debugTrajectorySOSv(" innermostTSOS of TransientTrack", 
			     alongTTr.innermostMeasurementState());
    this->debugTrajectorySOSv(" outermostTSOS of TransientTrack", 
			     alongTTr.outermostMeasurementState());
    this->debugTrajectorySOS(" initialTSOS for theFitter ", initialTSOS);
    std::cout<<" . . . . . trajVec.size() "<<trajVec.size()<<std::endl;
    if(trajVec.size()) this->debugTrajectory(" theFitter trajectory ", trajVec[0]);
  }

  if(!smooth){
    if(trajVec.size()) trackFittedTSOS = trajVec[0].lastMeasurement().updatedState();}
  else{
    std::vector<Trajectory> trajSVec;
    if (trajVec.size()){
      if(direction == alongMomentum) trajSVec = theSmoother->trajectories(*(trajVec.begin()));
      else                           trajSVec = theSmootherOp->trajectories(*(trajVec.begin()));
    }
    if(info) std::cout<<" . . . . trajSVec.size() "<<trajSVec.size()<<std::endl;
    if(trajSVec.size()) this->debugTrajectorySOSv("smoothed geom InnermostState",
						  trajSVec[0].geometricalInnermostState());
    if(trajSVec.size()) trackFittedTSOS = trajSVec[0].geometricalInnermostState();
  }

  if(info) this->debugTrajectorySOSv(" trackFittedTSOS ", trackFittedTSOS);

} // end of trackFitter

// ----  refit any direction of muon transient track ------
void 
GlobalTrackerMuonAlignment::muonFitter(reco::TrackRef alongTr, reco::TransientTrack& alongTTr,
				       PropagationDirection direction,
				       TrajectoryStateOnSurface& trackFittedTSOS)
{  
  bool info = false;
  bool smooth = false;
  
  if(info)  std::cout<<" ......... start of muonFitter ........"<<std::endl;  
  if(false && info){  
    this->debugTrackHit(" Muon track hits ", alongTr); 
    this->debugTrackHit(" Muon TransientTrack hits ", alongTTr); 
  }

  edm::OwnVector<TrackingRecHit> recHit;
  DetId trackDetId = DetId(alongTr->innerDetId());
  for (trackingRecHit_iterator i=alongTr->recHitsBegin();i!=alongTr->recHitsEnd(); i++) {
    recHit.push_back((*i)->clone());
  }
  if(direction != alongMomentum)
    {
      edm::OwnVector<TrackingRecHit> recHitAlong = recHit;
      recHit.clear();
      for(unsigned int ihit = recHitAlong.size()-1; ihit+1>0; ihit--){
	recHit.push_back(recHitAlong[ihit]);
      }
      recHitAlong.clear();
    }
  if(info)
    std::cout<<" ... Own recHit.size() DetId==Muon "<<recHit.size()<<" "<<trackDetId.rawId()<<std::endl;

  MuonTransientTrackingRecHitBuilder::ConstRecHitContainer recHitMu =
    MuRHBuilder->build(alongTTr.recHitsBegin(), alongTTr.recHitsEnd());
  if(info)
    std::cout<<" ...along.... recHitMu.size() "<<recHitMu.size()<<std::endl;
  if(direction != alongMomentum){
    MuonTransientTrackingRecHitBuilder::ConstRecHitContainer recHitMuAlong = recHitMu;
    recHitMu.clear();
    for(unsigned int ihit = recHitMuAlong.size()-1; ihit+1>0; ihit--){
      recHitMu.push_back(recHitMuAlong[ihit]);
    }
    recHitMuAlong.clear();
  }
  if(info)
    std::cout<<" ...opposite... recHitMu.size() "<<recHitMu.size()<<std::endl;

  TrajectoryStateOnSurface firstTSOS;
  if(direction == alongMomentum) firstTSOS = alongTTr.innermostMeasurementState();
  else                           firstTSOS = alongTTr.outermostMeasurementState();

  AlgebraicSymMatrix55 CovLoc;
  CovLoc(0,0) = 1.   * firstTSOS.localError().matrix()(0,0);
  CovLoc(1,1) = 10.  * firstTSOS.localError().matrix()(1,1);
  CovLoc(2,2) = 10.  * firstTSOS.localError().matrix()(2,2);
  CovLoc(3,3) = 100. * firstTSOS.localError().matrix()(3,3);
  CovLoc(4,4) = 100. * firstTSOS.localError().matrix()(4,4);
  TrajectoryStateOnSurface initialTSOS(firstTSOS.localParameters(), LocalTrajectoryError(CovLoc),
				       firstTSOS.surface(), &*magneticField_);
  
  PTrajectoryStateOnDet PTraj = 
    trajectoryStateTransform::persistentState(initialTSOS, trackDetId.rawId());
  //const TrajectorySeed seedT(*PTraj, recHit, alongMomentum);  
  const TrajectorySeed seedT(PTraj, recHit, direction);  

  std::vector<Trajectory> trajVec;
  if(direction == alongMomentum) trajVec = theFitter->fit(seedT, recHitMu, initialTSOS);
  else                           trajVec = theFitterOp->fit(seedT, recHitMu, initialTSOS);
  
  if(info){
    this->debugTrajectorySOSv(" innermostTSOS of TransientTrack", 
			     alongTTr.innermostMeasurementState());
    this->debugTrajectorySOSv(" outermostTSOS of TransientTrack", 
			     alongTTr.outermostMeasurementState());
    this->debugTrajectorySOS(" initialTSOS for theFitter ", initialTSOS);
    std::cout<<" . . . . . trajVec.size() "<<trajVec.size()<<std::endl;
    if(trajVec.size()) this->debugTrajectory(" theFitter trajectory ", trajVec[0]);
  }

  if(!smooth){
    if(trajVec.size()) trackFittedTSOS = trajVec[0].lastMeasurement().updatedState();}
  else{
    std::vector<Trajectory> trajSVec;
    if (trajVec.size()){
      if(direction == alongMomentum) trajSVec = theSmoother->trajectories(*(trajVec.begin()));
      else                           trajSVec = theSmootherOp->trajectories(*(trajVec.begin()));
    }
    if(info) std::cout<<" . . . . trajSVec.size() "<<trajSVec.size()<<std::endl;
    if(trajSVec.size()) this->debugTrajectorySOSv("smoothed geom InnermostState",
						  trajSVec[0].geometricalInnermostState());
    if(trajSVec.size()) trackFittedTSOS = trajSVec[0].geometricalInnermostState();
  }

} // end of muonFitter


// ----  debug printout of hits from TransientTrack  ------
void GlobalTrackerMuonAlignment::debugTrackHit(const std::string title, TransientTrack& alongTr) 
{
  std::cout<<" ------- "<<title<<" --------"<<std::endl;
  int nHit = 1;
  TransientTrackingRecHit::RecHitContainer recHit;
  for (trackingRecHit_iterator i=alongTr.recHitsBegin();i!=alongTr.recHitsEnd(); i++) {
    std::cout<<" Hit "<<nHit++<<" DetId "<<(*i)->geographicalId().det();
    if( (*i)->geographicalId().det() == DetId::Tracker ) std::cout<<" Tracker ";
    else if ( (*i)->geographicalId().det() == DetId::Muon ) std::cout<<" Muon "; 
    else std::cout<<" Unknown ";
    if(!(*i)->isValid()) std::cout<<" not valid "<<std::endl;  
    else std::cout<<std::endl;
  }
}


// ----  debug printout of hits from TrackRef   ------
void GlobalTrackerMuonAlignment::debugTrackHit(const std::string title, reco::TrackRef alongTr) 
{
  std::cout<<" ------- "<<title<<" --------"<<std::endl;
  int nHit = 1;
  TransientTrackingRecHit::RecHitContainer recHit;
  for (trackingRecHit_iterator i=alongTr->recHitsBegin();i!=alongTr->recHitsEnd(); i++) {
    std::cout<<" Hit "<<nHit++<<" DetId "<<(*i)->geographicalId().det();
    if( (*i)->geographicalId().det() == DetId::Tracker ) std::cout<<" Tracker ";
    else if ( (*i)->geographicalId().det() == DetId::Muon ) std::cout<<" Muon "; 
    else std::cout<<" Unknown ";
    if(!(*i)->isValid()) std::cout<<" not valid "<<std::endl;  
    else std::cout<<std::endl;
  }
}

// ----  debug printout TrajectoryStateOnSurface  ------
void GlobalTrackerMuonAlignment::debugTrajectorySOS(const std::string title, 
						     TrajectoryStateOnSurface& trajSOS) 
{
  std::cout<<"    --- "<<title<<" --- "<<std::endl;
  if(!trajSOS.isValid()) {std::cout<<"      Not valid !!!! "<<std::endl; return;}
  std::cout<<" R |p| "<<trajSOS.globalPosition().perp()<<" "
	   <<trajSOS.globalMomentum().mag()<<" charge "<<trajSOS.charge()<<std::endl; 
  std::cout<<" x p  "<<trajSOS.globalParameters().vector()(0)<<" "
	   <<trajSOS.globalParameters().vector()(1)<<" "
	   <<trajSOS.globalParameters().vector()(2)<<" "
	   <<trajSOS.globalParameters().vector()(3)<<" "
	   <<trajSOS.globalParameters().vector()(4)<<" "
	   <<trajSOS.globalParameters().vector()(5)<<std::endl;
  std::cout<<" +/- "<<sqrt(trajSOS.cartesianError().matrix()(0,0))<<" "
	   <<sqrt(trajSOS.cartesianError().matrix()(1,1))<<" "
	   <<sqrt(trajSOS.cartesianError().matrix()(2,2))<<" "
	   <<sqrt(trajSOS.cartesianError().matrix()(3,3))<<" "
	   <<sqrt(trajSOS.cartesianError().matrix()(4,4))<<" "
	   <<sqrt(trajSOS.cartesianError().matrix()(5,5))<<std::endl;
  std::cout<<" q/p dxy/dz xy "<<trajSOS.localParameters().vector()(0)<<" "
	   <<trajSOS.localParameters().vector()(1)<<" "
	   <<trajSOS.localParameters().vector()(2)<<" "
	   <<trajSOS.localParameters().vector()(3)<<" "
	   <<trajSOS.localParameters().vector()(4)<<std::endl;
  std::cout<<"   +/- error  "<<sqrt(trajSOS.localError().matrix()(0,0))<<" "
	   <<sqrt(trajSOS.localError().matrix()(1,1))<<" "
	   <<sqrt(trajSOS.localError().matrix()(2,2))<<" "
	   <<sqrt(trajSOS.localError().matrix()(3,3))<<" "
	   <<sqrt(trajSOS.localError().matrix()(4,4))<<" "<<std::endl;
}

// ----  debug printout TrajectoryStateOnSurface  ------
void GlobalTrackerMuonAlignment::debugTrajectorySOSv(const std::string title, 
						     TrajectoryStateOnSurface trajSOS) 
{
  std::cout<<"    --- "<<title<<" --- "<<std::endl;
  if(!trajSOS.isValid()) {std::cout<<"      Not valid !!!! "<<std::endl; return;}
  std::cout<<" R |p| "<<trajSOS.globalPosition().perp()<<" "
	   <<trajSOS.globalMomentum().mag()<<" charge "<<trajSOS.charge()<<std::endl; 
  std::cout<<" x p  "<<trajSOS.globalParameters().vector()(0)<<" "
	   <<trajSOS.globalParameters().vector()(1)<<" "
	   <<trajSOS.globalParameters().vector()(2)<<" "
	   <<trajSOS.globalParameters().vector()(3)<<" "
	   <<trajSOS.globalParameters().vector()(4)<<" "
	   <<trajSOS.globalParameters().vector()(5)<<std::endl;
  std::cout<<" +/- "<<sqrt(trajSOS.cartesianError().matrix()(0,0))<<" "
	   <<sqrt(trajSOS.cartesianError().matrix()(1,1))<<" "
	   <<sqrt(trajSOS.cartesianError().matrix()(2,2))<<" "
	   <<sqrt(trajSOS.cartesianError().matrix()(3,3))<<" "
	   <<sqrt(trajSOS.cartesianError().matrix()(4,4))<<" "
	   <<sqrt(trajSOS.cartesianError().matrix()(5,5))<<std::endl;
  std::cout<<" q/p dxy/dz xy "<<trajSOS.localParameters().vector()(0)<<" "
	   <<trajSOS.localParameters().vector()(1)<<" "
	   <<trajSOS.localParameters().vector()(2)<<" "
	   <<trajSOS.localParameters().vector()(3)<<" "
	   <<trajSOS.localParameters().vector()(4)<<std::endl;
  std::cout<<"   +/- error  "<<sqrt(trajSOS.localError().matrix()(0,0))<<" "
	   <<sqrt(trajSOS.localError().matrix()(1,1))<<" "
	   <<sqrt(trajSOS.localError().matrix()(2,2))<<" "
	   <<sqrt(trajSOS.localError().matrix()(3,3))<<" "
	   <<sqrt(trajSOS.localError().matrix()(4,4))<<" "<<std::endl;
}

// ----  debug printout Trajectory   ------
void GlobalTrackerMuonAlignment::debugTrajectory(const std::string title, Trajectory& traj) 
{
  std::cout<<"\n"<<"    ...... "<<title<<" ...... "<<std::endl;
  if(!traj.isValid()) {std::cout<<"          Not valid !!!!!!!!  "<<std::endl; return;}
  std::cout<<" chi2/Nhit "<<traj.chiSquared()<<" / "<<traj.foundHits();
  if(traj.direction() == alongMomentum) std::cout<<" alongMomentum  >>>>"<<std::endl;
  else                                  std::cout<<" oppositeToMomentum  <<<<"<<std::endl;
  this->debugTrajectorySOSv(" firstMeasurementTSOS ",traj.firstMeasurement().updatedState());
  //this->debugTrajectorySOSv(" firstMeasurementTSOS ",traj.firstMeasurement().predictedState());
  this->debugTrajectorySOSv(" lastMeasurementTSOS ",traj.lastMeasurement().updatedState());
  //this->debugTrajectorySOSv(" geom InnermostState", traj.geometricalInnermostState());
  std::cout<<"    . . . . . . . . . . . . . . . . . . . . . . . . . . . . \n"<<std::endl;
}


// ----  write GlobalPositionRcd   ------
void GlobalTrackerMuonAlignment::writeGlPosRcd(CLHEP::HepVector& paramVec) 
{
  //paramVec(1) = 0.1; paramVec(2) = 0.2; paramVec(3) = 0.3;         //0123
  //paramVec(4) = 0.0001; paramVec(5) = 0.0002; paramVec(6) = 0.0003;
  //paramVec(1) = 0.3; paramVec(2) = 0.4; paramVec(3) = 0.5;         //03450123
  //paramVec(4) = 0.0001; paramVec(5) = 0.0002; paramVec(6) = 0.0003;
  //paramVec(1) = 2.; paramVec(2) = 2.; paramVec(3) = 2.;               //222
  //paramVec(4) = 0.02; paramVec(5) = 0.02; paramVec(6) = 0.02;
  //paramVec(1) = -10.; paramVec(2) = -20.; paramVec(3) = -30.;         //102030
  //paramVec(4) = -0.1; paramVec(5) = -0.2; paramVec(6) = -0.3;
  //paramVec(1) = 0.; paramVec(2) = 0.; paramVec(3) = 1.;               //1z
  //paramVec(4) = 0.0000; paramVec(5) = 0.0000; paramVec(6) = 0.0000;

  std::cout<<" paramVector "<<paramVec.T()<<std::endl;
  
  CLHEP::Hep3Vector colX, colY, colZ;

#ifdef NOT_EXACT_ROTATION_MATRIX     
  colX = CLHEP::Hep3Vector(          1.,  -paramVec(6),  paramVec(5));
  colY = CLHEP::Hep3Vector( paramVec(6),            1., -paramVec(4));
  colZ = CLHEP::Hep3Vector(-paramVec(5),   paramVec(4),           1.);
#else 
  double s1 = std::sin(paramVec(4)), c1 = std::cos(paramVec(4));
  double s2 = std::sin(paramVec(5)), c2 = std::cos(paramVec(5));
  double s3 = std::sin(paramVec(6)), c3 = std::cos(paramVec(6));
  colX = CLHEP::Hep3Vector(               c2 * c3,               -c2 * s3,       s2);
  colY = CLHEP::Hep3Vector(c1 * s3 + s1 * s2 * c3, c1 * c3 - s1 * s2 * s3, -s1 * c2);
  colZ = CLHEP::Hep3Vector(s1 * s3 - c1 * s2 * c3, s1 * c3 + c1 * s2 * s3,  c1 * c2);
#endif

  CLHEP::HepVector  param0(6,0);

  Alignments* globalPositions = new Alignments();  

  //Tracker
  AlignTransform tracker(iteratorTrackerRcd->translation(),
			 iteratorTrackerRcd->rotation(),
			 DetId(DetId::Tracker).rawId());
  // Muon
  CLHEP::Hep3Vector posMuGlRcd = iteratorMuonRcd->translation();
  CLHEP::HepRotation rotMuGlRcd = iteratorMuonRcd->rotation();
  CLHEP::HepEulerAngles angMuGlRcd = iteratorMuonRcd->rotation().eulerAngles();
  if(debug_)
    std::cout<<" Old muon Rcd Euler angles "<<angMuGlRcd.phi()<<" "<<angMuGlRcd.theta()
	     <<" "<<angMuGlRcd.psi()<<std::endl;
  AlignTransform muon;
  if((angMuGlRcd.phi() == 0.) && (angMuGlRcd.theta() == 0.) && (angMuGlRcd.psi() == 0.) &&
     (posMuGlRcd.x() == 0.) && (posMuGlRcd.y() == 0.) && (posMuGlRcd.z() == 0.)){
    std::cout<<" New muon parameters are stored in Rcd"<<std::endl; 

    AlignTransform muonNew(AlignTransform::Translation(paramVec(1),
						       paramVec(2),
						       paramVec(3)),
			   AlignTransform::Rotation   (colX,
						       colY,
						       colZ),
			   DetId(DetId::Muon).rawId());
    muon = muonNew;
  }
  else if((paramVec(1) == 0.) && (paramVec(2) == 0.) && (paramVec(3) == 0.) && 
	  (paramVec(4) == 0.) && (paramVec(5) == 0.) && (paramVec(6) == 0.)){
    std::cout<<" Old muon parameters are stored in Rcd"<<std::endl; 
    
    AlignTransform muonNew(iteratorMuonRcd->translation(),
			   iteratorMuonRcd->rotation(),
			   DetId(DetId::Muon).rawId());
    muon = muonNew;  
  }
  else{
    std::cout<<" New + Old muon parameters are stored in Rcd"<<std::endl; 
    CLHEP::Hep3Vector posMuGlRcdThis = CLHEP::Hep3Vector(paramVec(1), paramVec(2), paramVec(3));   
    CLHEP::HepRotation rotMuGlRcdThis = CLHEP::HepRotation(colX, colY, colZ);   
    CLHEP::Hep3Vector posMuGlRcdNew = rotMuGlRcdThis * posMuGlRcd + posMuGlRcdThis;
    CLHEP::HepRotation rotMuGlRcdNew = rotMuGlRcdThis * rotMuGlRcd;

    AlignTransform muonNew(posMuGlRcdNew,
			   rotMuGlRcdNew, 
			   DetId(DetId::Muon).rawId());
    muon = muonNew;
  }

  // Ecal
  AlignTransform ecal(iteratorEcalRcd->translation(),
			 iteratorEcalRcd->rotation(),
			 DetId(DetId::Ecal).rawId());
  // Hcal
  AlignTransform hcal(iteratorHcalRcd->translation(),
			 iteratorHcalRcd->rotation(),
			 DetId(DetId::Hcal).rawId());
  // Calo
  AlignTransform calo(AlignTransform::Translation(param0(1),
						  param0(2),
						  param0(3)),
		      AlignTransform::EulerAngles(param0(4),
						  param0(5),
						  param0(6)),
		      DetId(DetId::Calo).rawId());

  std::cout << "Tracker (" << tracker.rawId() << ") at " << tracker.translation() 
	    << " " << tracker.rotation().eulerAngles() << std::endl;
  std::cout << tracker.rotation() << std::endl;
  
  std::cout << "Muon (" << muon.rawId() << ") at " << muon.translation() 
    	    << " " << muon.rotation().eulerAngles() << std::endl;
  std::cout << "          rotations angles around x,y,z "
	    << " ( " << -muon.rotation().zy() << " " << muon.rotation().zx() 
	    << " " << -muon.rotation().yx() << " )" << std::endl;
   std::cout << muon.rotation() << std::endl;
   
   std::cout << "Ecal (" << ecal.rawId() << ") at " << ecal.translation() 
	     << " " << ecal.rotation().eulerAngles() << std::endl;
   std::cout << ecal.rotation() << std::endl;
   
   std::cout << "Hcal (" << hcal.rawId() << ") at " << hcal.translation()
	     << " " << hcal.rotation().eulerAngles() << std::endl;
   std::cout << hcal.rotation() << std::endl;
   
   std::cout << "Calo (" << calo.rawId() << ") at " << calo.translation()
	     << " " << calo.rotation().eulerAngles() << std::endl;
   std::cout << calo.rotation() << std::endl;

   globalPositions->m_align.push_back(tracker);
   globalPositions->m_align.push_back(muon);
   globalPositions->m_align.push_back(ecal);
   globalPositions->m_align.push_back(hcal);
   globalPositions->m_align.push_back(calo);

   std::cout << "Uploading to the database..." << std::endl;

   edm::Service<cond::service::PoolDBOutputService> poolDbService;
   
   if (!poolDbService.isAvailable())
     throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
                  
   //    if (poolDbService->isNewTagRequest("GlobalPositionRcd")) {
//       poolDbService->createNewIOV<Alignments>(&(*globalPositions), poolDbService->endOfTime(), "GlobalPositionRcd");
//    } else {
//       poolDbService->appendSinceTime<Alignments>(&(*globalPositions), poolDbService->currentTime(), "GlobalPositionRcd");
//    }
   poolDbService->writeOne<Alignments>(&(*globalPositions), 
				       poolDbService->currentTime(),
				       //poolDbService->beginOfTime(),
                                       "GlobalPositionRcd");
   std::cout << "done!" << std::endl;
   
   return;
}


//define this as a plug-in
DEFINE_FWK_MODULE(GlobalTrackerMuonAlignment);
