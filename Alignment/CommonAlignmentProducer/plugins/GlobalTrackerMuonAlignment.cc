// -*- C++ -*-
//
// Package:    GlobalTrackerMuonAlignment
// Class:      GlobalTrackerMuonAlignment
// 
/**\class GlobalTrackerMuonAlignment GlobalTrackerMuonAlignment.cc Alignment/GlobalTrackerMuonAlignment/src/GlobalTrackerMuonAlignment.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alexandre Spiridonov
//         Created:  Fri Oct 16 15:59:05 CEST 2009
//
// $Id$
//

// system include files
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <fstream>

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
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

// products
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/SmartPropagator.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirectionChooser.h" 
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackPropagation/RungeKutta/interface/RKTestPropagator.h" 
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

// Database
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"

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

  //inline ESHandle<Propagator> propagator() const;

  void gradientGlobal(GlobalVector&, GlobalVector&, GlobalVector&, GlobalVector&, 
		      GlobalVector&, AlgebraicSymMatrix66&);
  void gradientGlobalAlg(GlobalVector&, GlobalVector&, GlobalVector&, GlobalVector&,  
  			 AlgebraicSymMatrix66&);
  void misalignMuon(GlobalVector&, GlobalVector&, GlobalVector&, GlobalVector&, 
		    GlobalVector&, GlobalVector&);
  void writeGlPosRcd(CLHEP::HepVector& d3); 

  inline double CLHEP_dot(CLHEP::HepVector a, CLHEP::HepVector b)
  {
    return a(1)*b(1)+a(2)*b(2)+a(3)*b(3);
  }

 private:

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------

  edm::InputTag trackTags_;     // used to select what tracks to read from configuration file
  edm::InputTag muonTags_;      // used to select what standalone muons
  edm::InputTag gmuonTags_;     // used to select what global muons
  edm::InputTag smuonTags_;     // used to select what selected muons
  string propagator_;           // name of the propagator
  bool cosmicMuonMode_;   
  bool isolatedMuonMode_;
  string rootOutFile_;
  string txtOutFile_;
  bool writeDB_;                // write results to DB
  bool debug_;                  // run in debugging mode

  edm::ESWatcher<GlobalTrackingGeometryRecord> watchTrackingGeometry_;
  const GlobalTrackingGeometry * trackingGeometry_;

  edm::ESWatcher<IdealMagneticFieldRecord> watchMagneticFieldRecord_;
  const MagneticField * magneticField_;

  edm::ESWatcher<TrackingComponentsRecord> watchTrackingComponentsRecord_;

  //                            // LSF for global d(3)    Inf * d = Gfr 
  AlgebraicVector3 Gfr;         // free terms 
  AlgebraicSymMatrix33 Inf;     // information matrix 

  int N_event;
  static const int NFit_d = 6;  // number of fitted parameters d(NFit_d)
  CLHEP::HepVector MuGlShift;   // expected global muon shifts  
  CLHEP::HepVector MuGlAngle;   // expected global muon angles  
  CLHEP::HepVector Grad;        // the gradient of the objective function
  CLHEP::HepMatrix Hess;        // the Hessian     -- // ---     
  char MuSelect[100];           // what part of muon system is selected for 1st hit 

  ofstream OutGlobalTxt;       // output the vector of global alignment as text     

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
   rootOutFile_(iConfig.getUntrackedParameter<string>("rootOutFile")),
   txtOutFile_(iConfig.getUntrackedParameter<string>("txtOutFile")),
   writeDB_(iConfig.getUntrackedParameter<bool>("writeDB")),
   debug_(iConfig.getUntrackedParameter<bool>("debug"))
{
  if (cosmicMuonMode_==false && isolatedMuonMode_==false) {
    throw cms::Exception("BadConfig") << "Operation mode not set! "
				      << "Please set either cosmics or isolated to true.";
  }
  
  if (cosmicMuonMode_==true && isolatedMuonMode_==true) {
    throw cms::Exception("BadConfig") << "Both cosmics and isolated mode are true! "
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
  using namespace edm;
  using namespace reco;  
  //using reco::TrackCollection;
  //using reco::MuonCollection;

  bool alarm = false;
  //bool alarm = true;
  double PI = 3.1415927;
 
  cosmicMuonMode_ = true; // true: both Cosmic and IsolatedMuon are treated with 1,2 tracks
  //cosmicMuonMode_ = false;
  isolatedMuonMode_ = !cosmicMuonMode_; //true: only IsolatedMuon are treated with 1 track
 
  N_event++;
  if (debug_) {
    std::cout << "-------- event " << N_event << " ----";
    if (cosmicMuonMode_) std::cout << " treated as CosmicMu ";
    if (isolatedMuonMode_) std::cout <<" treated as IsolatedMu ";
    std::cout << std::endl;
  }
  
  Handle<reco::TrackCollection> tracks;
  Handle<reco::TrackCollection> muons;
  Handle<reco::TrackCollection> gmuons;
  Handle<reco::MuonCollection> smuons;
  
  //iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "TrackerOnly",  tracks);
  //iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "StandAlone",  muons);
  //iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "GlobalMuon",  gmuons);
  //iEvent.getByLabel("ALCARECOMuAlCalIsolatedMu", "SelectedMuons",  smuons);
  
  iEvent.getByLabel(trackTags_,tracks);
  iEvent.getByLabel(muonTags_,muons);
  iEvent.getByLabel(gmuonTags_,gmuons);
  iEvent.getByLabel(smuonTags_,smuons);
  
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
      //std::cout<<" is GMu Mu SAMu TrMu "<<itMuon->isGlobalMuon()<<" "
      //       <<itMuon->isMuon()<<" "<<itMuon->isStandAloneMuon()
      //       <<" "<<itMuon->isTrackerMuon()<<" "
      //       <<std::endl;
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

 //std::cout<<" ========================= "<<propagator_<<std::endl;
  // from RPC efficiency
  
  ESHandle<Propagator> propagator;
  iSetup.get<TrackingComponentsRecord>().get(propagator_, propagator);
  
  SteppingHelixPropagator alongStHePr = 
    SteppingHelixPropagator(magneticField_, alongMomentum);  
  SteppingHelixPropagator oppositeStHePr = 
    SteppingHelixPropagator(magneticField_, oppositeToMomentum);  

  //double tolerance = 5.e-5;
  RKTestPropagator alongRKPr = 
    RKTestPropagator(magneticField_, alongMomentum);
  RKTestPropagator oppositeRKPr = 
    RKTestPropagator(magneticField_, oppositeToMomentum);

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

    if(false & debug_){
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
    if(debug_)
      std::cout<<" passed lenght momentum cuts"<<std::endl;

    GlobalVector GRm, GPm, Nl, Rm, Pm, Rt, Pt, Rt0;
    AlgebraicSymMatrix66 Cm, C0, Ce, C1; 
    TrajectoryStateOnSurface extrapolationT;
    
    if( isolatedMuonMode_ ){      //------------------------------- Isolated Muon -----
      const Surface& refSurface = innerMuTSOS.surface(); 
      ReferenceCountingPointer<TangentPlane> 
	tpMuLocal(refSurface.tangentPlane(innerMuTSOS.localPosition()));
      Nl = tpMuLocal->normalVector();
      ReferenceCountingPointer<TangentPlane> 
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
	ReferenceCountingPointer<TangentPlane> 
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
	ReferenceCountingPointer<TangentPlane> 
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
	ReferenceCountingPointer<TangentPlane> 
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
    
    GlobalTrackerMuonAlignment::misalignMuon(GRm, GPm, Nl, Rt, Rm, Pm);
    //Rm = Rt;
    //Pm = Pt;

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

    if(Pt.mag() < 15.) continue;
    if(Pm.mag() < 10.) continue;
    if(fabs(RelMomResidual) > 0.5) continue;
    if(fabs(resR.x()) > 20.) continue;
    if(fabs(resR.y()) > 20.) continue;
    if(fabs(resR.z()) > 20.) continue;
    if(fabs(resR.mag()) > 30.) continue;
    if(fabs(resP.x()) > 0.05) continue;
    if(fabs(resP.y()) > 0.05) continue;
    if(fabs(resP.z()) > 0.05) continue;
    if(chi_d > 40.) continue;

    //if(sqrt(C0(0,0) > 0.005 )) continue;
    //if(sqrt(C1(0,0)) < 0.0001 || sqrt(C1(0,0)) > 0.03 ) continue;
    //if(!(fabs(Nl.x()) > 0.4 & fabs(Nl.x()) < 0.6)) continue; 
    
    //                                            select Barrel 
    //if(Rmuon < 400. || Rmuon > 450.) continue; 
    //if(Zmuon < -600. || Zmuon > 600.) continue;
    //if(fabs(Nl.z()) > 0.95) continue;  
    //std::sprintf(MuSelect, "    Barrel");
    //                                                  EndCap1
    //if(Rmuon < 120. || Rmuon > 450.) continue;
    //if(Zmuon < -720.) continue;
    //if(Zmuon > -580.) continue;
    //if(fabs(Nl.z()) < 0.95) continue;  
    //std::sprintf(MuSelect, "    EndCap1");
    //                                                  EndCap2
    //if(Rmuon < 120. || Rmuon > 450.) continue;
    //if(Zmuon >  720.) continue;
    //if(Zmuon <  580.) continue;
    //if(fabs(Nl.z()) < 0.95) continue;  
    //std::sprintf(MuSelect, "    EndCap2");
    //                                                 select All
    if(Rmuon < 120. || Rmuon > 450.) continue;  
    if(Zmuon < -720. || Zmuon > 720.) continue;
    std::sprintf(MuSelect, "      Barrel+EndCaps");

    if(debug_)
      std::cout<<" .............. passed all cuts"<<std::endl;
        
    //                     gradient and Hessian for each track
    
    GlobalTrackerMuonAlignment::gradientGlobalAlg(Rt, Pt, Rm, Nl, Cm);
    GlobalTrackerMuonAlignment::gradientGlobal(Rt, Pt, Rm, Pm, Nl, Cm);
    
    
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
    
    histo18->Fill(sqrt(C0(0,0))); 
    histo19->Fill(sqrt(C1(0,0))); 
    histo20->Fill(sqrt(C1(0,0)+Ce(0,0))); 		   
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
    
    if (false & debug_) {   //--------------------------------- debug print ----------
      
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
      
      static AlgebraicSymMatrix66 Ro;
      double Diag[5];
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
}

// ------------ method called once for job just before starting event loop  ------------
void 
GlobalTrackerMuonAlignment::beginJob()
{
  N_event = 0;
  double PI = 3.1415927;

  for(int i=0; i<=2; i++){
    Gfr(i) = 0.;
    for(int j=0; j<=2; j++){
      Inf(i,j) = 0.;
    }}

  int Nd = NFit_d;
  Grad =  CLHEP::HepVector(Nd,0);
  Hess =  CLHEP::HepSymMatrix(Nd,0);

  TDirectory * dirsave = gDirectory;

  file = new TFile(rootOutFile_.c_str(), "recreate"); 
  const bool oldAddDir = TH1::AddDirectoryStatus(); 

  TH1::AddDirectory(true);
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
  histo11 = new TH1F("distance muon-track","distance muon w.r.t track [cm]",100,0.,20.);
  histo11->GetXaxis()->SetTitle("distance of muon w.r.t. track [cm]");
  histo12 = new TH1F("Xmuon-Xtrack","Xmuon-Xtrack [cm]",100,-20.,20.);
  histo12->GetXaxis()->SetTitle("Xmuon - Xtrack [cm]"); 
  histo13 = new TH1F("Ymuon-Ytrack","Ymuon-Ytrack [cm]",100,-20.,20.);
  histo13->GetXaxis()->SetTitle("Ymuon - Ytrack [cm]");
  histo14 = new TH1F("Zmuon-Ztrack","Zmuon-Ztrack [cm]",100,-20.,20.);       
  histo14->GetXaxis()->SetTitle("Zmuon-Ztrack [cm]");
  histo15 = new TH1F("NXmuon-NXtrack","NXmuon-NXtrack [rad]",100,-.1,.1);       
  histo15->GetXaxis()->SetTitle("N_{X}(muon)-N_{X}(track) [rad]");
  histo16 = new TH1F("NYmuon-NYtrack","NYmuon-NYtrack [rad]",100,-.1,.1);       
  histo16->GetXaxis()->SetTitle("N_{Y}(muon)-N_{Y}(track) [rad]");
  histo17 = new TH1F("NZmuon-NZtrack","NZmuon-NZtrack [rad]",100,-.1,.1);       
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

  histo101 = new TH2F("Rtr/mu vs Ztr/mu","hit of track/muon",100,-800.,800.,100,0.,600.);
  histo101->GetXaxis()->SetTitle("Z of track/muon [cm]");
  histo101->GetYaxis()->SetTitle("R of track/muon [cm]");
  histo102 = new TH2F("Ytr/mu vs Xtr/mu","hit of track/muon",100,-600.,600.,100,-600.,600.);
  histo102->GetXaxis()->SetTitle("X of track/muon [cm]");
  histo102->GetYaxis()->SetTitle("Y of track/muon [cm]");

  TH1::AddDirectory(oldAddDir);

  dirsave->cd();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GlobalTrackerMuonAlignment::endJob() {
  bool alarm = false;

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
  //                                   -------------- end of Global Algebraic

  
  if(debug_){
    std::cout<<"  inversed Inf "<<std::endl;
    std::cout<<InfI<<std::endl;
    std::cout<<" Gfr "<<Gfr<<std::endl;
    std::cout<<" -- Inf --"<<std::endl;
    std::cout<<Inf<<std::endl;
  }
    
  std::cout<<" ---- "<<N_event<<" event  "<<MuSelect<<" ---- "<<std::endl;
  std::cout<<smuonTags_<<std::endl;
  std::cout<<" Similated shifts[cm] "
	   <<MuGlShift(1)<<" "<<MuGlShift(2)<<" "<<MuGlShift(3)<<" "
	   <<" angles[rad] "<<MuGlAngle(1)<<" "<<MuGlAngle(2)<<" "<<MuGlAngle(3)<<" "
	   <<std::endl;
  std::cout<<" d    "<<d<<std::endl;;
  std::cout<<" +-   "<<sqrt(InfI(0,0))<<" "<<sqrt(InfI(1,1))<<" "<<sqrt(InfI(2,2))
	   <<std::endl;
  
  //                                ---------------  alignment Global CLHEP

  CLHEP::HepVector d3 = CLHEP::solve(Hess, -Grad);
  int iEr3;
  CLHEP::HepMatrix Errd3 = Hess.inverse(iEr3);
  if( iEr3 != 0) { if(alarm)
    std::cout<<" grad3 Error inverse  Hess matrix !!!!!!!!!!!"<<std::endl;
  }
  //                                ---------------- and of Global CLHEP

  if(NFit_d == 3){
    std::cout<<" d3   "<<d3(1)<<" "<<d3(2)<<" "<<d3(3)
	     <<std::endl;;
    std::cout<<" +-   "<<sqrt(Errd3(1,1))<<" "<<sqrt(Errd3(2,2))<<" "<<sqrt(Errd3(3,3))
	     <<std::endl;
  }
  if(NFit_d == 6){
    std::cout<<" d3   "<<d3(1)<<" "<<d3(2)<<" "<<d3(3)<<" "<<d3(4)<<" "
	     <<d3(5)<<" "<<d3(6)<<std::endl;;
    std::cout<<" +-   "<<sqrt(Errd3(1,1))<<" "<<sqrt(Errd3(2,2))<<" "<<sqrt(Errd3(3,3))
	     <<" "<<sqrt(Errd3(4,4))<<" "<<sqrt(Errd3(5,5))<<" "<<sqrt(Errd3(6,6))
	     <<std::endl;
  }

  // write GlobalPositionRcd
  if (writeDB_) 
    GlobalTrackerMuonAlignment::writeGlPosRcd(d3);

  // write global parameters to text file 
  OutGlobalTxt.open(txtOutFile_.c_str(), ios::out);
  if(!OutGlobalTxt.is_open()) std::cout<<" outglobal.txt is not open !!!!!"<<std::endl;
  else{
    OutGlobalTxt<<d3(1)<<" "<<d3(2)<<" "<<d3(3)<<" "<<d3(4)<<" "
		<<d3(5)<<" "<<d3(6)<<" muon Global.\n";
    OutGlobalTxt<<sqrt(Errd3(1,1))<<" "<<sqrt(Errd3(2,2))<<" "<<sqrt(Errd3(3,3))
		<<" "<<sqrt(Errd3(4,4))<<" "<<sqrt(Errd3(5,5))<<" "<<sqrt(Errd3(6,6))
		<<" errors.\n";
    OutGlobalTxt<<N_event<<" events are processed.\n";
    OutGlobalTxt<<smuonTags_<<".\n";
    OutGlobalTxt.close();
    std::cout<<" Write to the file outglobal.txt done  "<<std::endl;
  }

  file->Write(); 
  file->Close(); 
}

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
    Wi(i) = 1./Cm(i,i);
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

// ----  calculate gradient and Hessian matrix to search global parameters ------
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
  if( NFit_d != 6 ) {
    if(debug_)
      std::cout<<" !!!!!!!!!   Wrong NFit_d = "<<NFit_d<<std::endl;
    return;}

  int Nd = NFit_d;     // dimension of vector of alignment pararmeters, d 

  //  transform  momentum  --> direction,   i.e. Pm , Pt   ---> Nm, Nt 
  //double PtMom = GPt.mag();
  CLHEP::HepSymMatrix w(Nd,0);
  for (int i=1; i <= Nd; i++)
    for (int j=1; j <= Nd; j++){
      if(j <= i ) w(i,j) = GCov(i-1, j-1);
      //if(i >= 3) w(i,j) /= PtMom;
      //if(j >= 3) w(i,j) /= PtMom;
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
      std::cout<<" gradient3: inversion of matrix w fail "<<std::endl;
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
      std::cout<<" grad3 Error inverse  Hess matrix !!!!!!!!!!!"<<std::endl;
  }

  if(debug_){
    std::cout<<" d3   "<<d3I(1)<<" "<<d3I(2)<<" "<<d3I(3)<<" "<<d3I(4)<<" "
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
}

// ----  perform a misalignment of muon system   ------
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
  //d(1)=10.; d(2)=20.; d(3)=30.; a(1)=0.0100; a(2)=0.0200; a(3)=0.0300; // huge      
  //d(1)=10.; d(2)=20.; d(3)=30.; a(1)=0.2000; a(2)=0.2500; a(3)=0.3000; // huge angles      
  //d(1)=0.3; d(2)=0.4; d(3)=0.5; a(1)=0.0005; a(2)=0.0006; a(3)=0.0007; // 0345 0567
  //d(1)=0.3; d(2)=0.4; d(3)=0.5; a(1)=0.0001; a(2)=0.0002; a(3)=0.0003; // 0345 0123

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
}

// ----  write GlobalPositionRcd   ------
void GlobalTrackerMuonAlignment::writeGlPosRcd(CLHEP::HepVector& paramVec) 
{
  std::cout<<" paramVector "<<paramVec.T()<<std::endl;

  CLHEP::HepVector  param0(6,0);

  Alignments* globalPositions = new Alignments();

  AlignTransform tracker(AlignTransform::Translation(param0(1),
						     param0(2),
						     param0(3)),
			 AlignTransform::EulerAngles(param0(4),
						     param0(5),
						     param0(6)),
			 DetId(DetId::Tracker).rawId());
  AlignTransform muon(AlignTransform::Translation(paramVec(1),
						  paramVec(2),
						  paramVec(3)),
		      AlignTransform::EulerAngles(paramVec(4),
						  paramVec(5),
						  paramVec(6)),
		      DetId(DetId::Muon).rawId());
  AlignTransform ecal(AlignTransform::Translation(param0(1),
						  param0(2),
						  param0(3)),
		      AlignTransform::EulerAngles(param0(4),
						  param0(5),
						  param0(6)),
		      DetId(DetId::Ecal).rawId());
  AlignTransform hcal(AlignTransform::Translation(param0(1),
						  param0(2),
						  param0(3)),
		      AlignTransform::EulerAngles(param0(4),
						  param0(5),
						  param0(6)),
		      DetId(DetId::Hcal).rawId());
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
