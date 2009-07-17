/**\class TkLasBeamFitter TkLasBeamFitter.cc Alignment/LaserAlignment/plugins/TkLasBeamFitter.cc

  Original Authors:  Gero Flucke/Kolja Kaschube
           Created:  Wed May  6 08:43:02 CEST 2009
           $Id: TkLasBeamFitter.cc,v 1.1 2009/05/11 10:01:28 flucke Exp $

 Description: Fitting LAS beams with track model and providing TrajectoryStateOnSurface for hits.

 Implementation:
    - TkLasBeamCollection read from edm::Run
    - currently all done in beginRun(..),
      but should move to endRun(..) to allow a correct sequence with 
      production of TkLasBeamCollection in LaserAlignment::endRun(..)
*/


// system include files
#include <memory>

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>
#include <Geometry/Records/interface/TrackerDigiGeometryRecord.h>
#include <MagneticField/Engine/interface/MagneticField.h>
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

// data formats
// for edm::InRun
#include "DataFormats/Provenance/interface/BranchType.h"
// laser data formats
#include "DataFormats/LaserAlignment/interface/TkLasBeam.h"
#include "DataFormats/LaserAlignment/interface/TkFittedLasBeam.h"

// further includes
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Alignment/LaserAlignment/interface/TsosVectorCollection.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <iostream>
#include "TMinuit.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"

using namespace edm;
using namespace std;


//
// class declaration
//

class TkLasBeamFitter : public edm::EDProducer {
public:
  explicit TkLasBeamFitter(const edm::ParameterSet &config);
  ~TkLasBeamFitter();
  
  //virtual void beginJob(const edm::EventSetup& /*access deprecated*/) {}
  virtual void produce(edm::Event &event, const edm::EventSetup &setup);
  virtual void beginRun(edm::Run &run, const edm::EventSetup &setup);
  virtual void endRun(edm::Run &run, const edm::EventSetup &setup);
  //virtual void endRun(edm::Event &run, const edm::EventSetup &setup);
  //virtual void endJob() {}

private:
  /// Fit 'beam' using info from its base class TkLasBeam and set its parameters.
  /// Also fill 'tsoses' with TSOS for each LAS hit. 
  void getLasBeams(TkLasBeam &beam, TkFittedLasBeam &fittedBeam);
  void getLasHits(TkLasBeam &beam); 
  void fillVectors(TkLasBeam &beam);
  static double tecPlusFunction(double *x, double *par);
  static double tecMinusFunction(double *x, double *par);
  static double atFunction(double *x, double *par);
  void fitter(TkLasBeam &beam, AlgebraicSymMatrix &covMatrix);
  void trackPhi(TkLasBeam &beam);
  void globalTrackPoint(TkLasBeam &beam, 
			std::vector<double> &trackPhi, std::vector<double> &trackPhiRef);
  void buildTrajectory(TkLasBeam &beam);
  bool fitBeam(TkLasBeam &beam, TkFittedLasBeam &fittedBeam,
	       AlgebraicSymMatrix &covMatrix);

  // ----------member data ---------------------------
  const edm::InputTag src_;

  static vector<const GeomDetUnit*> gd;
  static vector<GlobalPoint> globHit;
  static vector<double> gLocalHitError;
  static vector<GlobalPoint> globPTrack;
  static vector<GlobalPoint> globPref;
  static vector<LocalPoint> gLasHit;
  static vector<double> gHitPhi;
  static vector<double> gHitPhiError;
  static vector<double> gHitZ;
  static vector<double> gHitZprime;
  static vector<double> gHitZprimeError;
  static vector<double> gBarrelModuleRadius;
  static vector<double> gBarrelModuleOffset;

  static vector<double> gFitPhi;
  static vector<double> gFitPhiError;
  static vector<double> gFitZprime;
  static vector<double> gFitZprimeError;

  static double gBeamR4;
  static double gBeamR6;
  static double gBeamR;
  static double gBeamZ0;
  static double gBeamZBS;
  static double gBeamZmin;
  static double gBeamZmax;
  static double gBeamSplitterZprime;
  static int gInvalidHits;
  static int gInvalidHitsAtTecMinus;
  static int atCounter;

  static unsigned int nFitParams;
  // fit parameters
  static double gAtMinusParam;
  static double gAtMinusParamError;
  static double gAtPlusParam;
  static double gAtPlusParamError;
//   static double gPhiAtMinusParam;
//   static double gPhiAtMinusParamError;
//   static double gThetaAtMinusParam;
//   static double gThetaAtMinusParamError;
//   static double gPhiAtPlusParam;
//   static double gPhiAtPlusParamError;
//   static double gThetaAtPlusParam;
//   static double gThetaAtPlusParamError;
  static double gSlope;
  static double gOffset;
  static double gSlopeError;
  static double gOffsetError;
  static double gBeamSplitterAngleParam;
  static double gBeamSplitterAngleParamError;

  static double chi2;
  static vector<TrajectoryStateOnSurface> gTsosLas;

  TFile *out;
  // create histos
  // filled in getLasBeams(..)
  TH1F *h_bsAngle;
  TH2F *h_bsAngleVsBeam;
  // filled in getLasHits(..)
  TH1F *h_hitX;
  TH1F *h_hitXTecPlus;
  TH1F *h_hitXTecMinus;
  TH1F *h_hitXAt;
  TH2F *h_hitXvsZTecPlus;
  TH2F *h_hitXvsZTecMinus;
  TH2F *h_hitXvsZAt;
  // filled in makeLasTracks(..)
  TH1F *h_chi2;
  TH1F *h_chi2ndof;
  TH1F *h_pull;
  TH1F *h_res;
  TH1F *h_resTecPlus;
  TH1F *h_resTecMinus;
  TH1F *h_resAt;
  TH2F *h_resVsZTecPlus;
  TH2F *h_resVsZTecMinus;
  TH2F *h_resVsZAt;
  TH2F *h_resVsHitTecPlus;
  TH2F *h_resVsHitTecMinus;
  TH2F *h_resVsHitAt;
}; 

//
// constants, enums and typedefs
//


//
// static data member definitions
//

vector<const GeomDetUnit*> TkLasBeamFitter::gd;
vector<GlobalPoint> TkLasBeamFitter::globHit;
vector<double> TkLasBeamFitter::gLocalHitError;
vector<GlobalPoint> TkLasBeamFitter::globPTrack;
vector<GlobalPoint> TkLasBeamFitter::globPref;
vector<LocalPoint> TkLasBeamFitter::gLasHit;
vector<double> TkLasBeamFitter::gHitPhi;
vector<double> TkLasBeamFitter::gHitPhiError;
vector<double> TkLasBeamFitter::gHitZ;
vector<double> TkLasBeamFitter::gHitZprime;
vector<double> TkLasBeamFitter::gHitZprimeError;
vector<double> TkLasBeamFitter::gBarrelModuleRadius;
vector<double> TkLasBeamFitter::gBarrelModuleOffset;

vector<double> TkLasBeamFitter::gFitPhi;
vector<double> TkLasBeamFitter::gFitPhiError;
vector<double> TkLasBeamFitter::gFitZprime;
vector<double> TkLasBeamFitter::gFitZprimeError;

double TkLasBeamFitter::gBeamR4 = 56.4; // real value!
double TkLasBeamFitter::gBeamR6 = 84.0; // real value!
double TkLasBeamFitter::gBeamR = 0.0;
double TkLasBeamFitter::gBeamZ0 = 0.0; 
double TkLasBeamFitter::gBeamZmin = 0.0;
double TkLasBeamFitter::gBeamZmax = 0.0;
double TkLasBeamFitter::gBeamSplitterZprime = 0.0;
int TkLasBeamFitter::gInvalidHits = 0;
int TkLasBeamFitter::gInvalidHitsAtTecMinus = 0;
int TkLasBeamFitter::atCounter = 0;

unsigned int TkLasBeamFitter::nFitParams = 0;
// fit parameters
double TkLasBeamFitter::gAtMinusParam = 0.0;
double TkLasBeamFitter::gAtMinusParamError = 0.0;
double TkLasBeamFitter::gAtPlusParam = 0.0;
double TkLasBeamFitter::gAtPlusParamError = 0.0;
// double TkLasBeamFitter::gPhiAtMinusParam = 0.0;
// double TkLasBeamFitter::gPhiAtMinusParamError = 0.0;
// double TkLasBeamFitter::gThetaAtMinusParam = 0.0;
// double TkLasBeamFitter::gThetaAtMinusParamError = 0.0;
// double TkLasBeamFitter::gPhiAtPlusParam = 0.0;
// double TkLasBeamFitter::gPhiAtPlusParamError = 0.0;
// double TkLasBeamFitter::gThetaAtPlusParam = 0.0;
// double TkLasBeamFitter::gThetaAtPlusParamError = 0.0;
double TkLasBeamFitter::gSlope = 0.0;
double TkLasBeamFitter::gOffset = 0.0;
double TkLasBeamFitter::gSlopeError = 0.0;
double TkLasBeamFitter::gOffsetError = 0.0;
double TkLasBeamFitter::gBeamSplitterAngleParam = 0.0;
double TkLasBeamFitter::gBeamSplitterAngleParamError = 0.0;

double TkLasBeamFitter::chi2 = 0.0;
vector<TrajectoryStateOnSurface> TkLasBeamFitter::gTsosLas;

// handles
Handle<TkLasBeamCollection> laserBeams;
ESHandle<MagneticField> fieldHandle;
ESHandle<TrackerGeometry> geometry;

//
// constructors and destructor
//
TkLasBeamFitter::TkLasBeamFitter(const edm::ParameterSet &iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src"))
{
  // declare the products to produce
  this->produces<TkFittedLasBeamCollection, edm::InRun>();
  this->produces<TsosVectorCollection, edm::InRun>();
  
  //now do what ever other initialization is needed
}

//---------------------------------------------------------------------------------------
TkLasBeamFitter::~TkLasBeamFitter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

//---------------------------------------------------------------------------------------
// ------------ method called to produce the data  ------------
void TkLasBeamFitter::produce(edm::Event &iEvent, const edm::EventSetup &setup)
{
  // Nothing per event!
}

//---------------------------------------------------------------------------------------
// ------------ method called at end of each run  ---------------------------------------
void TkLasBeamFitter::endRun(edm::Run &run, const edm::EventSetup &setup)
{
}
// FIXME!
// Indeed, that should be in endRun(..) - as soon as AlignmentProducer can call
// the algorithm's endRun correctly!


void TkLasBeamFitter::beginRun(edm::Run &run, const edm::EventSetup &setup)
{

//   edm::Service<TFileService> fileService;
  out = new TFile("residuals.root","UPDATE");
  // create histos of residuals
  // filled in getLasHits(..)
  h_hitX = new TH1F("hitX","local x of LAS hits;local x [cm];N",100,-0.5,0.5);
  h_hitXTecPlus = new TH1F("hitXTecPlus","local x of LAS hits in TECplus;local x [cm];N",100,-0.5,0.5);
  h_hitXTecMinus = new TH1F("hitXTecMinus","local x of LAS hits in TECminus;local x [cm];N",100,-0.5,0.5);
  h_hitXAt = new TH1F("hitXAt","local x of LAS hits in ATs;local x [cm];N",100,-2.5,2.5);
  h_hitXvsZTecPlus = new TH2F("hitXvsZTecPlus","local x vs z in TECplus;z [cm];local x [cm]",80,120,280,100,-0.5,0.5);
  h_hitXvsZTecMinus = new TH2F("hitXvsZTecMinus","local x vs z in TECMinus;z [cm];local x [cm]",80,-280,-120,100,-0.5,0.5);
  h_hitXvsZAt = new TH2F("hitXvsZAt","local x vs z in ATs;z [cm];local x [cm]",200,-200,200,100,-0.5,0.5);
  // filled in makeLasTracks(..)
  h_chi2 = new TH1F("chi2","#chi^{2};#chi^{2};N",100,0,1000);
  h_chi2ndof = new TH1F("chi2ndof","#chi^{2} per degree of freedom;#chi^{2}/N_{dof};N",100,0,100);
  h_pull = new TH1F("pull","pulls of #phi residuals;pull;N",50,-10,10);
  h_res = new TH1F("res","#phi residuals;#phi_{track} - #phi_{hit} [rad];N",60,-0.0015,0.0015);
  h_resTecPlus = new TH1F("resTecPlus","#phi residuals in TECplus;#phi_{track} - #phi_{hit} [rad];N",30,-0.0015,0.0015);
  h_resTecMinus = new TH1F("resTecMinus","#phi residuals in TECminus;#phi_{track} - #phi_{hit} [rad];N",60,-0.0015,0.0015);
  h_resAt = new TH1F("resAt","#phi residuals in ATs;#phi_{track} - #phi_{hit} [rad];N",30,-0.0015,0.0015);
  h_resVsZTecPlus = new TH2F("resVsZTecPlus","phi residuals vs. z in TECplus;z [cm];#phi_{track} - #phi_{hit} [rad]",
			 80,120,280,100,-0.0015,0.0015);
  h_resVsZTecMinus = new TH2F("resVsZTecMinus","phi residuals vs. z in TECminus;z [cm];#phi_{track} - #phi_{hit} [rad]",
			  80,-280,-120,100,-0.0015,0.0015);
  h_resVsZAt = new TH2F("resVsZAt","#phi residuals vs. z in ATs;N;#phi_{track} - #phi_{hit} [rad]",
		    200,-200,200,100,-0.0015,0.0015);
  h_resVsHitTecPlus = new TH2F("resVsHitTecPlus","#phi residuals vs. hits in TECplus;hit no.;#phi_{track} - #phi_{hit} [rad]",
			   144,0,144,100,-0.0015,0.0015);
  h_resVsHitTecMinus = new TH2F("resVsHitTecMinus","#phi residuals vs. hits in TECminus;hit no.;#phi_{track} - #phi_{hit} [rad]",
			    144,0,144,100,-0.0015,0.0015);
  h_resVsHitAt = new TH2F("resVsHitAt","#phi residuals vs. hits in ATs;hit no.;#phi_{track} - #phi_{hit} [rad]",
		      176,0,176,100,-0.0015,0.0015);
  // filled in getLasBeams(..)
  h_bsAngle = new TH1F("bsAngle","fitted beam splitter angle;BS angle [rad];N",40,-0.004,0.004);
  h_bsAngleVsBeam = new TH2F("bsAngleVsBeam","fitted beam splitter angle per beam;Beam no.;BS angle [rad]",
			 40,0,300,100,-0.004,0.004);

  // Create output collections - they are parallel.
  // (edm::Ref etc. and thus edm::AssociationVector are not supported for edm::Run...)
  std::auto_ptr<TkFittedLasBeamCollection> fittedBeams(new TkFittedLasBeamCollection);
  // One std::vector<TSOS> for each TkFittedLasBeam:
  std::auto_ptr<TsosVectorCollection> tsosesVec(new TsosVectorCollection);

  // get TkLasBeams, Tracker geometry, magnetic field
  run.getByLabel( "LaserAlignment", "tkLaserBeams", laserBeams );
  setup.get<TrackerDigiGeometryRecord>().get(geometry);
  setup.get<IdealMagneticFieldRecord>().get(fieldHandle);

  // loop over LAS beams
  for(TkLasBeamCollection::const_iterator iBeam = laserBeams->begin(), iEnd = laserBeams->end();
       iBeam != iEnd; ++iBeam){

    TkLasBeam beam = TkLasBeam(*iBeam);
    TkFittedLasBeam fittedBeam = TkFittedLasBeam(*iBeam);
    
    gTsosLas.clear();

    // call main function; all other functions are called inside getLasBeams(..)
    this->getLasBeams(beam, fittedBeam);
    
    // fill output products
    fittedBeams->push_back(fittedBeam);

    cout << "No. of TSOS in beam: " << gTsosLas.size() << endl;
    tsosesVec->push_back(gTsosLas);

//     if(!this->fitBeam(fittedBeams->back(), tsosesVec->back())){
//       edm::LogError("BadFit") 
// 	 << "Problems fitting TkLasBeam, id " << fittedBeams->back().getBeamId() << ".";
//       fittedBeams->pop_back(); // remove last entry added just before
//       tsosesVec->pop_back();   // dito
//     }
  }
  
  // finally, put fitted beams and TSOS vectors into run
  run.put(fittedBeams);
  run.put(tsosesVec);

  cout << "TkLasBeamFitter done. Saving and cleaning up..." << endl;

  // save histos and clean up
  h_hitX->Write();
  h_hitXTecPlus->Write();
  h_hitXTecMinus->Write();
  h_hitXAt->Write();
  h_hitXvsZTecPlus->Write();
  h_hitXvsZTecMinus->Write();
  h_hitXvsZAt->Write();
  h_res->Write();
  h_resTecPlus->Write();
  h_resTecMinus->Write();
  h_resAt->Write();
  h_resVsZTecPlus->Write();
  h_resVsZTecMinus->Write();
  h_resVsZAt->Write();
  h_resVsHitTecPlus->Write();
  h_resVsHitTecMinus->Write();
  h_resVsHitAt->Write();
  h_bsAngle->Write();
  h_bsAngleVsBeam->Write();
  h_pull->Write();
  h_chi2->Write();
  h_chi2ndof->Write();
  out->Close();

  delete h_hitX;
  delete h_hitXTecPlus;
  delete h_hitXTecMinus;
  delete h_hitXAt;
  delete h_hitXvsZTecPlus;
  delete h_hitXvsZTecMinus;
  delete h_hitXvsZAt;
  delete h_res;
  delete h_resTecPlus;
  delete h_resTecMinus;
  delete h_resAt;
  delete h_resVsZTecPlus;
  delete h_resVsZTecMinus;
  delete h_resVsZAt;
  delete h_resVsHitTecPlus;
  delete h_resVsHitTecMinus;
  delete h_resVsHitAt;
  delete h_bsAngle;
  delete h_bsAngleVsBeam;
  delete h_pull;
  delete h_chi2;
  delete h_chi2ndof;
  delete out;
}


// methods for las data processing

// -------------- loop over beams, call functions ----------------------------
void TkLasBeamFitter::getLasBeams(TkLasBeam &beam, TkFittedLasBeam &fittedBeam)
{
  cout << "---------------------------------------" << endl;
  cout << "beam id: " << beam.getBeamId() << " isTec: " << (beam.isTecInternal() ? "Y" : "N") 
       << " isTec+: " << (beam.isTecInternal(1) ? "Y" : "N") << " isTec-: " << (beam.isTecInternal(-1) ? "Y" : "N")
       << " isAt: " << (beam.isAlignmentTube() ? "Y" : "N") << " isR6: " << (beam.isRing6() ? "Y" : "N") << endl;
      
  // only use good beams -> adjust according to data!
  if(beam.getBeamId() != 0 && beam.getBeamId() != 30 && beam.getBeamId() != 50 &&
     beam.getBeamId() != 100 && beam.getBeamId() != 130 && beam.getBeamId() != 150 ){
    
    // set right beam radius
    gBeamR = beam.isRing6() ? gBeamR6 : gBeamR4;
    cout << "nominal beam radius = " << gBeamR << endl;
    
    // get LAS hits
    this->getLasHits(beam);
    
    // set z values for fit
    double sumZ = 0;
    for(int hit = 0; hit < static_cast<int>(globHit.size()); hit++){
      sumZ += globHit[hit].z();
    }
    gBeamZ0 = sumZ / (globHit.size() - gInvalidHits);
    
    // TECplus
    if(beam.isTecInternal(1)){
      gBeamSplitterZprime = 205.75 - gBeamZ0;
      gBeamZmin = 120.0 - gBeamZ0;
      gBeamZmax = 280.0 - gBeamZ0;
    }
    // TECminus
    else if(beam.isTecInternal(-1)){
      gBeamSplitterZprime = -205.75 - gBeamZ0;
      gBeamZmin = -280.0 - gBeamZ0;
      gBeamZmax = -120.0 - gBeamZ0;
    }
    // AT
    else{
      gBeamSplitterZprime = 112.3 - gBeamZ0;
      gBeamZmin = -200.0 - gBeamZ0;
      gBeamZmax = 200.0 - gBeamZ0;
    }
    cout << "z0 = " << gBeamZ0 << ", z_bs' = " << gBeamSplitterZprime 
	 << ", zmin = " << gBeamZmin << ", zmax = " << gBeamZmax << endl;
    
    // fill vectors for fit
    this->fillVectors(beam);
    
    // do fit, build tracks
    if(gHitZprime.size() > 0){
      
      // number of fit parameters
      unsigned int tecParams = 3;
      unsigned int atParams = 3;
      // ATs: if no TEC hits, no BS fit
      if(gInvalidHitsAtTecMinus == 5) atParams = atParams - 1;
      if(beam.isTecInternal(1)){
	nFitParams = tecParams;
      }
      else if(beam.isTecInternal(-1)){
	nFitParams = tecParams;
      }
      else{
	nFitParams = atParams;
      }
      
      AlgebraicSymMatrix covMatrix(nFitParams, 1);
      this->fitter(beam, covMatrix);
      
      this->trackPhi(beam);
      
      this->buildTrajectory(beam);
      this->fitBeam(beam, fittedBeam, covMatrix);
      
      // fill histos
      
      // include slope, offset, covariance plots here
      
      if(gBeamSplitterAngleParam != 0){
	h_bsAngle->Fill(2.0*atan(0.5*gBeamSplitterAngleParam));
	h_bsAngleVsBeam->Fill(beam.getBeamId(), 2.0*atan(0.5*gBeamSplitterAngleParam));
      }
    }
    else cout << "no hit, no fit" << endl;
  }
}

// --------- get hits, convert to global coordinates ---------------------------
void TkLasBeamFitter::getLasHits(TkLasBeam &beam)
{
  gd.clear();
  globHit.clear();
  gLasHit.clear();
  gInvalidHits = 0;
  gInvalidHitsAtTecMinus = 0;
  gBeamSplitterAngleParam = 0.0;

  for( TkLasBeam::const_iterator iHit = beam.begin(); iHit < beam.end(); ++iHit ){
    // iHit is a SiStripLaserRecHit2D
    cout << "Strip hit (local x): " << iHit->localPosition().x() << " +- " << iHit->localPositionError().xx() << endl;
 
    // get global position of LAS hits
    gd.push_back(geometry->idToDetUnit(iHit->getDetId()));
    gLasHit.push_back(iHit->localPosition());
    GlobalPoint globPtemp(gd.back()->toGlobal(gLasHit.back()));

    // bad hits, beams get dummy global position
    if(iHit->localPosition().x() == 0.0 || iHit->localPositionError().xx() > 0.1){
      globHit.push_back(GlobalPoint(0,0,0));
      gInvalidHits ++;
      cout << "invalid hit" << endl;
      // ATs: TECminus invalid hits need counting
      if(beam.isAlignmentTube() && globPtemp.z() < -112.3){
	gInvalidHitsAtTecMinus ++;
      }
    }
    // bad beams in CRAFT08 data! (Overlapping AT and TEC modules are ignored.)
    else if((beam.getBeamId() == 200 && abs(globPtemp.z()) > 112.3) ||
	    (beam.getBeamId() == 230 && abs(globPtemp.z()) > 112.3) ||
	    (beam.getBeamId() == 250 && abs(globPtemp.z()) > 112.3)
	    ){
      globHit.push_back(GlobalPoint(0,0,0));
      gInvalidHits ++;
      cout << "overlap with TEC beam: ignored!" << endl;
      if(beam.isAlignmentTube() && globPtemp.z() < -112.3){
	gInvalidHitsAtTecMinus ++;
      }
    }
    // now good hits
    else{ 
      // TECs
      if(beam.isTecInternal()){
	globHit.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, globPtemp.phi(), globPtemp.z())));
      }
      // ATs
      else{
	// TECs
	if(abs(globPtemp.z()) > 112.3){
	  globHit.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, globPtemp.phi(), globPtemp.z())));
	}
	// Barrel
	else{
	  globHit.push_back(globPtemp);
	}
      }
      // local position error
      gLocalHitError.push_back(iHit->localPositionError().xx());
      
      // fill histos
      h_hitX->Fill(iHit->localPosition().x());
      // TECplus
      if(beam.isTecInternal(1)){
	h_hitXTecPlus->Fill(iHit->localPosition().x());
	h_hitXvsZTecPlus->Fill(globHit.back().z(),iHit->localPosition().x());
      }
      // TECminus
      else if(beam.isTecInternal(-1)){
	h_hitXTecMinus->Fill(iHit->localPosition().x());
	h_hitXvsZTecMinus->Fill(globHit.back().z(),iHit->localPosition().x());
      }
      // ATs
      else{
	h_hitXAt->Fill(iHit->localPosition().x());
	h_hitXvsZAt->Fill(globHit.back().z(),iHit->localPosition().x());
      }
    }
  }
}


// ------------ fill vectors for fitted quantities -------------------
void TkLasBeamFitter::fillVectors(TkLasBeam &beam)
{
  gHitPhi.clear();
  gHitPhiError.clear();
  gHitZ.clear();
  gHitZprime.clear();
  gHitZprimeError.clear();
  gFitZprime.clear();
  gFitZprimeError.clear();
  gFitPhi.clear();
  gFitPhiError.clear();
  gBarrelModuleRadius.clear();
  gBarrelModuleOffset.clear();
  
  for(int hit = 0; hit < static_cast<int>(globHit.size()); ++hit){
    // valid hits
    if(globHit[hit].x() != 0.0){
      gHitPhi.push_back(static_cast<double>(globHit[hit].phi()));
      gHitPhiError.push_back( 0.003 / globHit[hit].perp()); // gLocalPositionError[hit] or assume 0.003
      gHitZ.push_back(globHit[hit].z());
      gHitZprime.push_back(globHit[hit].z() - gBeamZ0);
      gHitZprimeError.push_back(0.0);
      // now for use in fitter(..), only valid hits!
      gFitPhi.push_back(static_cast<double>(globHit[hit].phi()));
      gFitPhiError.push_back( 0.003 / globHit[hit].perp()); // gLocalPositionError[hit] or assume 0.003
      gFitZprime.push_back(globHit[hit].z() - gBeamZ0);
      gFitZprimeError.push_back(0.0);
      // barrel-specific values
      if(beam.isAlignmentTube() && abs(globHit[hit].z()) < 112.3){
	gBarrelModuleRadius.push_back(globHit[hit].perp());
	gBarrelModuleOffset.push_back(abs(gBarrelModuleRadius.back() - gBeamR));
      }
    }
    // invalid hits
    else{
      gHitPhi.push_back(0.0);
      gHitPhiError.push_back(0.0);
      gHitZ.push_back(0.0);
      gHitZprime.push_back(0.0);
    }
  }
}


// ------------ parametrization functions for las beam fits ------------
double TkLasBeamFitter::tecPlusFunction(double *x, double *par)
{
  double z = x[0]; // 'primed'? -> yes!!!

  if(z < gBeamSplitterZprime){
    return par[0] + par[1] * z;
  } 
  else{
    // par[2] = 2*tan(BeamSplitterAngle/2.0)
    return par[0] + par[1] * z - par[2] * (z - gBeamSplitterZprime)/gBeamR; 
  }
}

double TkLasBeamFitter::tecMinusFunction(double *x, double *par)
{
  double z = x[0]; // 'primed'? -> yes!!!

  if(z > gBeamSplitterZprime){
    return par[0] + par[1] * z;
  } 
  else{
    // par[2] = 2*tan(BeamSplitterAngle/2.0)
    return par[0] + par[1] * z + par[2] * (z - gBeamSplitterZprime)/gBeamR; 
  }
}
 
double TkLasBeamFitter::atFunction(double *x, double *par)
{
  double z = x[0]; // 'primed'? -> yes!!!
  // TECminus
  if(z < -gBeamSplitterZprime - 2.0*gBeamZ0){
    return par[0] + par[1] * z;
  }

  else if(-gBeamSplitterZprime - 2.0*gBeamZ0 < z && z < gBeamSplitterZprime){
  // TIB
  // par[3] = tan(PhiAtMinus), par[5] = tan(2.0*ThetaAtMinus)
  // par[4] = tan(PhiAtPlus), par[6] = tan(2.0*ThetaAtPlus)
    if(gFitZprime[5-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[0])
	+ gBarrelModuleOffset[0]/gBarrelModuleRadius[0] * (0); // -par[3] - par[5]
    }
    else if(gFitZprime[6-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[1])
	+ gBarrelModuleOffset[1]/gBarrelModuleRadius[1] * (0); // -par[3] - par[5]
    }
    else if(gFitZprime[7-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[2])
	+ gBarrelModuleOffset[2]/gBarrelModuleRadius[2] * (0); // -par[3] - par[5]
    }
    else if(gFitZprime[8-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[3])
	+ gBarrelModuleOffset[3]/gBarrelModuleRadius[3] * (0); // -par[4] - par[6]
    }
    else if(gFitZprime[9-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[4])
	+ gBarrelModuleOffset[4]/gBarrelModuleRadius[4] * (0); // -par[4] - par[6]
    }
    else if(gFitZprime[10-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[5])
	+ gBarrelModuleOffset[5]/gBarrelModuleRadius[5] * (0); // -par[4] - par[6]
    }
    // TOB
    else if(gFitZprime[11-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[6])
	+ gBarrelModuleOffset[6]/gBarrelModuleRadius[6] * (0); // par[3] - par[5]
    }
    else if(gFitZprime[12-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[7])
	+ gBarrelModuleOffset[7]/gBarrelModuleRadius[7] * (0); // par[3] - par[5]
    }
    else if(gFitZprime[13-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[8])
	+ gBarrelModuleOffset[8]/gBarrelModuleRadius[8] * (0); // par[3] - par[5]
    }
    else if(gFitZprime[14-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[9])
	+ gBarrelModuleOffset[9]/gBarrelModuleRadius[9] * (0); // par[4] - par[6]
    }
    else if(gFitZprime[15-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[10])
	+ gBarrelModuleOffset[10]/gBarrelModuleRadius[10] * (0); // par[4] - par[6]
    }
    else if(gFitZprime[16-gInvalidHitsAtTecMinus]){
      return par[0] + par[1] * (z - gBarrelModuleOffset[11])
	+ gBarrelModuleOffset[11]/gBarrelModuleRadius[11] * (0); // par[4] - par[6]
    }
    else return -99999.9;
  }
  // TECplus
  else{
    // par[2] = 2*tan(BeamSplitterAngle/2.0)
    return par[0] + par[1] * z - par[2] * (z - gBeamSplitterZprime)/gBeamR;
  }
}

// ------------ perform fit of beams ------------------------------------
void TkLasBeamFitter::fitter(TkLasBeam &beam, AlgebraicSymMatrix &covMatrix)
{
  cout << "used hits: " << gFitZprime.size() << ", invalid hits: " << gInvalidHits 
       << ", nBarrelModules: " << gBarrelModuleRadius.size() 
       << ", invalid hits in AT TECminus: " << gInvalidHitsAtTecMinus << endl;
  
  TGraphErrors *lasData = new TGraphErrors(gFitZprime.size(), 
					   &(gFitZprime[0]), &(gFitPhi[0]), 
					   &(gFitZprimeError[0]), &(gFitPhiError[0]));
  
  TF1 *tecPlus = new TF1("tecPlus", tecPlusFunction, gBeamZmin, gBeamZmax, nFitParams );
  TF1 *tecMinus = new TF1("tecMinus", tecMinusFunction, gBeamZmin, gBeamZmax, nFitParams );
  TF1 *at = new TF1("at", atFunction, gBeamZmin, gBeamZmax, nFitParams ); 

  // do fit (R = entire range)
  if(beam.isTecInternal(1)){
    lasData->Fit("tecPlus", "RV");
  }
  else if(beam.isTecInternal(-1)){
    lasData->Fit("tecMinus", "RV");
  }
  else{
    lasData->Fit("at","RV");
  }
  
  // get values and errors for offset and slope
  gMinuit->GetParameter(0, gOffset, gOffsetError);
  gMinuit->GetParameter(1, gSlope, gSlopeError);
  // ...and BS angle
  if(gInvalidHitsAtTecMinus != 5){
    gMinuit->GetParameter(2, gBeamSplitterAngleParam, gBeamSplitterAngleParamError);
  }
  else{
    gBeamSplitterAngleParam = 0.0;
    gBeamSplitterAngleParamError = 0.0;
  }

//   if(beam.isAlignmentTube()){
//     gMinuit->GetParameter(3, gAtMinusParam, gAtMinusParamError);
//     gMinuit->GetParameter(4, gAtPlusParam, gAtPlusParamError);
//     gMinuit->GetParameter(5, gPhiAtPlusParam, gPhiAtPlusParamError);
//     gMinuit->GetParameter(6, gThetaAtPlusParam, gThetaAtPlusParamError);
    
//     cout << "AtPlusParam = " << gAtPlusParam << " +- " << gAtPlusParamError 
// 	 << ", AtMinusParam = " << gAtMinusParam << " +- " << gAtMinusParamError << endl;

//     cout << "PhiAtMinus = " << atan(gPhiAtMinusParam) << ", PhiAtMinusParamError = " << gPhiAtMinusParamError
// 	 << ", ThetaAtMinus = " << 2.0*atan(gThetaAtMinusParam) << ", ThetaAtMinusParamError = " << gThetaAtMinusParamError
// 	 << ", PhiAtPlus = " << atan(gPhiAtPlusParam) << ", PhiAtPlusParamError = " << gPhiAtPlusParamError
// 	 << ", ThetaAtPlus = " << 2.0*atan(gThetaAtPlusParam) << ", ThetaAtPlusParamError = " << gThetaAtPlusParamError << endl;
//   }

  // get covariance matrix
  double corr01 = 0.0;
  double corr02 = 0.0;
  double corr12 = 0.0;
  vector<double> vec(nFitParams*nFitParams);
  gMinuit->mnemat(&vec[0], nFitParams);
  // fill covariance matrix
  for(unsigned int col = 0; col < nFitParams; col++){
    for(unsigned int row = 0; row < nFitParams; row++){
      covMatrix[col][row] = vec[row + nFitParams*col];
    }
  }
  // compute correlation between parameters
  corr01 = covMatrix[1][0]/(gOffsetError*gSlopeError);
  if(gInvalidHitsAtTecMinus != 5){
    corr02 = covMatrix[2][0]/(gOffsetError*gBeamSplitterAngleParamError);
    corr12 = covMatrix[2][1]/(gSlopeError*gBeamSplitterAngleParamError);
  }

  // display fit results
  cout << "number of parameters: " << nFitParams << endl
       << "a = " << gOffset << ", aError = " << gOffsetError << endl 
       << "b = " << gSlope  << ", bError = " << gSlopeError << endl;
  cout << "c = " << gBeamSplitterAngleParam << ", cError = " << gBeamSplitterAngleParamError << endl 
       << "BSangle = " << 2.0*atan(0.5*gBeamSplitterAngleParam) << ", BSangleError = " 
       << gBeamSplitterAngleParamError/(1+gBeamSplitterAngleParam*gBeamSplitterAngleParam/4.0) << endl;
  cout << "correlations: " << corr01 << ", " << corr02 << ", " << corr12 << endl;

  // clean up
  delete tecPlus;
  delete tecMinus;
  delete at;
  delete lasData;
}


// -------------- calculate track phi value ----------------------------------
void TkLasBeamFitter::trackPhi(TkLasBeam &beam)
{
  vector<double> trackPhi;
  // additional track point for trajectory calculation
  vector<double> trackPhiRef;

  // loop over LAS hits
  for(int hit = 0; hit < static_cast<int>(gHitZprime.size()); ++hit){
    // invalid hits
    if(gHitZprime[hit] == 0.0){
      trackPhi.push_back(0.0);
      trackPhiRef.push_back(0.0);
    }
    // good hits
    else{
      // TECplus
      if(beam.isTecInternal(1)){
	if(gHitZprime[hit] < gBeamSplitterZprime){
	  trackPhi.push_back(gOffset + gSlope * gHitZprime[hit]);
	  trackPhiRef.push_back(gOffset + gSlope * (gHitZprime[hit] + 1.0));
	}
	else{
	  trackPhi.push_back(gOffset + gSlope * gHitZprime[hit] 
			     - gBeamSplitterAngleParam * (gHitZprime[hit] - gBeamSplitterZprime)/gBeamR);
	  trackPhiRef.push_back(gOffset + gSlope * (gHitZprime[hit] + 1.0)
				- gBeamSplitterAngleParam * ((gHitZprime[hit] + 1.0) - gBeamSplitterZprime)/gBeamR);
	}
      }
      // TECminus
      else if(beam.isTecInternal(-1)){
	if(gHitZprime[hit] > gBeamSplitterZprime){
	  trackPhi.push_back(gOffset + gSlope * gHitZprime[hit]);
	  trackPhiRef.push_back(gOffset + gSlope * (gHitZprime[hit] + 1.0));
	}
	else{
	  trackPhi.push_back(gOffset + gSlope * gHitZprime[hit] 
			     + gBeamSplitterAngleParam * (gHitZprime[hit] - gBeamSplitterZprime)/gBeamR);
	  trackPhiRef.push_back(gOffset + gSlope * (gHitZprime[hit] + 1.0)
				+ gBeamSplitterAngleParam * ((gHitZprime[hit] + 1.0) - gBeamSplitterZprime)/gBeamR);
	}
      }
      // ATs
      else{
	// TECminus
	if(gHitZprime[hit] < -gBeamSplitterZprime - 2.0*gBeamZ0){
	  trackPhi.push_back(gOffset + gSlope * gHitZprime[hit]);
	  trackPhiRef.push_back(gOffset + gSlope * (gHitZprime[hit] + 1.0));
	}
	// BarrelMinus
	else if(-gBeamSplitterZprime - 2.0*gBeamZ0 < gHitZprime[hit] && gHitZprime[hit] < -gBeamZ0){
	  trackPhiRef.push_back(gOffset + gSlope * gHitZprime[hit]);
	  // TIB
	  if(gBarrelModuleOffset[hit-5] > 3.5){
	    cout << "module offset = " << gBarrelModuleOffset[hit-5] 
		 << ", radius = " << gBarrelModuleRadius[hit-5]<< endl;
	    trackPhi.push_back(gOffset + gSlope * (gHitZprime[hit] - gBarrelModuleOffset[hit-5])
			       + gBarrelModuleOffset[hit-5] / 
			       gBarrelModuleRadius[hit-5] * (0)); // check params!!!
	  }
	  // TOB
	  else{
	    cout << "module offset = " << gBarrelModuleOffset[hit-5] 
		 << ", radius = " << gBarrelModuleRadius[hit-5]<< endl;
	    trackPhi.push_back(gOffset + gSlope * (gHitZprime[hit] - gBarrelModuleOffset[hit-5])
			       + gBarrelModuleOffset[hit-5] / 
			       gBarrelModuleRadius[hit-5] * (0)); // check params!!!
	  }
	}
	// BarrelPlus
	else if(-gBeamZ0 < gHitZprime[hit] && gHitZprime[hit] < gBeamSplitterZprime){
	  trackPhiRef.push_back(gOffset + gSlope * gHitZprime[hit]);
	  // TIB
	  if(gBarrelModuleOffset[hit-5] > 3.5){
	    cout << "module offset = " << gBarrelModuleOffset[hit-5] 
		 << ", radius = " << gBarrelModuleRadius[hit-5] << endl;
	    trackPhi.push_back(gOffset + gSlope * (gHitZprime[hit] - gBarrelModuleOffset[hit-5])
			       + gBarrelModuleOffset[hit-5] / 
			       gBarrelModuleRadius[hit-5] * (0)); // check params!!!
	  }
	  // TOB
	  else{
	    cout << "module offset = " << gBarrelModuleOffset[hit-5] 
		 << ", radius = " << gBarrelModuleRadius[hit-5]<< endl;
	    trackPhi.push_back(gOffset + gSlope * (gHitZprime[hit] - gBarrelModuleOffset[hit-5])
			       + gBarrelModuleOffset[hit-5] / 
			       gBarrelModuleRadius[hit-5] * (0)); // check params!!!
	  }
	}
	// TECplus
	else{
	  trackPhi.push_back(gOffset + gSlope * gHitZprime[hit] 
			     - gBeamSplitterAngleParam * (gHitZprime[hit] - gBeamSplitterZprime)/gBeamR);
	  trackPhiRef.push_back(gOffset + gSlope * (gHitZprime[hit] + 1.0)
				- gBeamSplitterAngleParam * ((gHitZprime[hit] + 1.0) - gBeamSplitterZprime)/gBeamR);
	}
      }
    }
  }
  // calculate global position of track impact points
  this->globalTrackPoint(beam, trackPhi, trackPhiRef);
}


// -------------- calculate global track points, hit residuals, chi2 ----------------------------------
void TkLasBeamFitter::globalTrackPoint(TkLasBeam &beam, vector<double> &trackPhi, vector<double> &trackPhiRef)
{
  globPTrack.clear();
  globPref.clear();
  chi2 = 0.0;

  double phiResidual;
  double phiResidualPull = 0.0;

  // loop over LAS hits
  for(int hit = 0; hit < static_cast<int>(gHitZprime.size()); ++hit){
    // invalid hits
    if(trackPhi[hit] == 0.0 && trackPhiRef[hit] == 0.0){
      globPTrack.push_back(GlobalPoint(0,0,0));
      globPref.push_back(GlobalPoint(0,0,0));
    }
    // good hits
    else{
      // TECs
      if(beam.isTecInternal(0)){
	globPTrack.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhi[hit], gHitZ[hit])));
	globPref.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhiRef[hit], gHitZ[hit] + 1.0)));
      }
      // ATs
      else{
	// TECminus
	if(gHitZprime[hit] < -gBeamSplitterZprime - 2.0*gBeamZ0){
	  globPTrack.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhi[hit], gHitZ[hit])));
	  globPref.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhiRef[hit], gHitZ[hit] + 1.0)));
	}
	// TECplus
	else if(gHitZprime[hit] > gBeamSplitterZprime){
	  globPTrack.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhi[hit], gHitZ[hit])));
	  globPref.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhiRef[hit], gHitZ[hit] + 1.0)));
	}
	// Barrel
	else{
	  globPTrack.push_back(GlobalPoint(GlobalPoint::Cylindrical(globHit[hit].perp(), trackPhi[hit], gHitZ[hit])));
	  globPref.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhiRef[hit], gHitZ[hit])));
	}
      }

      cout << "hit (global): " << globHit[hit] << " r = " << globHit[hit].perp() << " phi = " << globHit[hit].phi() << endl
	   << "  fitted hit: " << globPTrack[hit] << " r = " << globPTrack[hit].perp() << ", phi = " << globPTrack[hit].phi() << endl
	   << "    reference point: " << globPref[hit] << " r = " << globPref[hit].perp() << ", phi = " << globPref[hit].phi() << endl;
      
      // calculate residuals = pred - hit (in global phi)
      phiResidual = globPTrack[hit].phi() - globHit[hit].phi();
      // pull calculation (FIX!!!)
      phiResidualPull = phiResidual / gHitPhiError[hit];
      //       sqrt(gHitPhiError[hit]*gHitPhiError[hit] + 
      // 	   (gOffsetError*gOffsetError + globPTrack[hit].z()*globPTrack[hit].z() * gSlopeError*gSlopeError));
      cout << "      phi residual = " << phiResidual << " +- " << gHitPhiError[hit] << ", pull = " << phiResidualPull << endl;
      // calculate chi2
      chi2 += phiResidual*phiResidual / (gHitPhiError[hit]*gHitPhiError[hit]);
      
      // fill histos
      h_res->Fill(phiResidual);
      // TECplus
      if(beam.isTecInternal(1)){
	h_pull->Fill(phiResidualPull);
	h_resTecPlus->Fill(phiResidual);
	h_resVsZTecPlus->Fill(globPTrack[hit].z(), phiResidual);
	// Ring 6
	if(beam.isRing6()){
	  h_resVsHitTecPlus->Fill(hit+(beam.getBeamId()-1)/10*9+72, phiResidual);
	}
	// Ring 4
	else{
	  h_resVsHitTecPlus->Fill(hit+beam.getBeamId()/10*9, phiResidual);
	}
      }
      // TECminus
      else if(beam.isTecInternal(-1)){
	h_pull->Fill(phiResidualPull);
	h_resTecMinus->Fill(phiResidual);
	h_resVsZTecMinus->Fill(globPTrack[hit].z(), phiResidual);
	// Ring 6
	if(beam.isRing6()){
	  h_resVsHitTecMinus->Fill(hit+(beam.getBeamId()-101)/10*9+72, phiResidual);
	}
	// Ring 4
	else{
	  h_resVsHitTecMinus->Fill(hit+(beam.getBeamId()-100)/10*9, phiResidual);
	}
      }
      // ATs
      else{
	h_pull->Fill(phiResidualPull);
	h_resAt->Fill(phiResidual);
	h_resVsZAt->Fill(globPTrack[hit].z(), phiResidual);
	h_resVsHitAt->Fill(hit+(beam.getBeamId()-200)/10*22, phiResidual);
      }
    }
  }
  cout << "chi^2 = " << chi2 << ", chi^2/ndof = " << chi2/(gHitZprime.size() - nFitParams - gInvalidHits) << endl;
  h_chi2->Fill(chi2);
  h_chi2ndof->Fill(chi2/(gHitZprime.size() - nFitParams - gInvalidHits));
}


// ----------- create TrajectoryStateOnSurface for each track hit ----------------------------------------------
void TkLasBeamFitter::buildTrajectory(TkLasBeam &beam)
{
  const MagneticField* magneticField = fieldHandle.product();

  for(int hit = 0; hit < static_cast<int>(globPTrack.size()); ++hit){

    GlobalVector trajectoryState;

    if(globPTrack[hit].x() == 0.0){
      // for invalid hit, fill with invalid TSOS
      gTsosLas.push_back(TrajectoryStateOnSurface());
    }
    else{
      // TECplus
      if(beam.isTecInternal(1)){
	trajectoryState = GlobalVector(globPref[hit]-globPTrack[hit]);
      }
      // TECminus
      else if(beam.isTecInternal(-1)){
	trajectoryState = GlobalVector(globPTrack[hit]-globPref[hit]);
      }
      // ATs
      else{
	// TECminus
	if(gHitZprime[hit] < -gBeamSplitterZprime - 2.0*gBeamZ0){
	  trajectoryState = GlobalVector(globPTrack[hit]-globPref[hit]);
	}
	// TECplus
	else if(gHitZprime[hit] > gBeamSplitterZprime){
	  trajectoryState = GlobalVector(globPref[hit]-globPTrack[hit]);
	}
	// Barrel
	else{
	  trajectoryState = GlobalVector(globPTrack[hit]-globPref[hit]);
	}
      }				
      const FreeTrajectoryState ftsLas = FreeTrajectoryState(globPTrack[hit],trajectoryState,0,magneticField);
      gTsosLas.push_back(TrajectoryStateOnSurface(ftsLas,gd[hit]->surface(),beforeSurface));
    }
  }
}


//---------------------- set beam parameters for fittedBeams ---------------------------------
bool TkLasBeamFitter::fitBeam(TkLasBeam &beam, TkFittedLasBeam &fittedBeam, AlgebraicSymMatrix &covMatrix)
{
  // set beam parameters for fittedBeam output
  unsigned int paramType = 0;

  std::vector<TkFittedLasBeam::Scalar> params(nFitParams); // two local, one global; replace number with parameter
  params[0] = gOffset;
  params[1] = gSlope;
  if(gInvalidHitsAtTecMinus != 5){
    params[2] = gBeamSplitterAngleParam;
  }

  AlgebraicMatrix derivatives(gTsosLas.size(), nFitParams);
  // fill derivatives matrix with local track derivatives
  for(unsigned int hit = 0; hit < gTsosLas.size(); ++hit){
    
    // d(delta phi)/d(gOffset) is same for every hit;
    derivatives[hit][0] = 1.0;

    // d(delta phi)/d(gSlope) and d(delta phi)/d(gBeamSplitterAngleParam) depend on parametrizations
    // TECplus
    if(beam.isTecInternal(1)){
      derivatives[hit][1] = globPTrack[hit].z();
      if(gHitZprime[hit] < gBeamSplitterZprime){
	derivatives[hit][2] = 0.0;
      }
      else{
	derivatives[hit][2] = - (globPTrack[hit].z() - gBeamSplitterZprime) / gBeamR;
      }
    }
    // TECminus
    else if(beam.isTecInternal(-1)){
      derivatives[hit][1] = globPTrack[hit].z();
      if(gHitZprime[hit] > gBeamSplitterZprime){
	derivatives[hit][2] = 0.0;
      }
      else{
	derivatives[hit][2] = (globPTrack[hit].z() - gBeamSplitterZprime) / gBeamR;
      }
    }
    // ATs
    else{
      // TECminus
      if(gHitZprime[hit] < -gBeamSplitterZprime - 2.0*gBeamZ0){
	derivatives[hit][1] = globPTrack[hit].z();
	if(gInvalidHitsAtTecMinus != 5){
	  derivatives[hit][2] = 0.0;
	}
      }
      // TECplus
      else if(gHitZprime[hit] > gBeamSplitterZprime){
	derivatives[hit][1] = globPTrack[hit].z();
	if(gInvalidHitsAtTecMinus != 5){
	  derivatives[hit][2] = - (globPTrack[hit].z() - gBeamSplitterZprime) / gBeamR;
	}
      }
      // Barrel
      else{
	derivatives[hit][1] = globPTrack[hit].z() - gBarrelModuleOffset[hit-5];
	if(gInvalidHitsAtTecMinus != 5){
	  derivatives[hit][2] = 0.0;
	}
      }
    }
  }

  unsigned int firstFixedParam = 3; // 3 parameters, but 0 and 1 local, while 2 is global/fixed

  // set fit results
  fittedBeam.setParameters(paramType, params, covMatrix, derivatives, firstFixedParam, chi2);

  return true; // return false in case of problems
}


//---------------------------------------------------------------------------------------
//define this as a plug-in
DEFINE_FWK_MODULE(TkLasBeamFitter);
