/**\class TkLasBeamFitter TkLasBeamFitter.cc Alignment/LaserAlignment/plugins/TkLasBeamFitter.cc

  Original Authors:  Gero Flucke/Kolja Kaschube
           Created:  Wed May  6 08:43:02 CEST 2009
           $Id: TkLasBeamFitter.cc,v 1.6 2009/08/31 10:26:50 kaschube Exp $

 Description: Fitting LAS beams with track model and providing TrajectoryStateOnSurface for hits.

 Implementation:
    - TkLasBeamCollection read from edm::Run
    - all done in endRun(..) to allow a correct sequence with 
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
#include "FWCore/ServiceRegistry/interface/Service.h"
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
#include "DataFormats/Alignment/interface/TkLasBeam.h"
#include "DataFormats/Alignment/interface/TkFittedLasBeam.h"

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
  // virtual void beginRun(edm::Run &run, const edm::EventSetup &setup);
  virtual void endRun(edm::Run &run, const edm::EventSetup &setup);
  //virtual void endJob() {}

private:
  /// Fit 'beam' using info from its base class TkLasBeam and set its parameters.
  /// Also fill 'tsoses' with TSOS for each LAS hit. 
  void getLasBeams(TkFittedLasBeam &beam,vector<TrajectoryStateOnSurface> &tsosLas);
  void getLasHits(TkFittedLasBeam &beam, SiStripLaserRecHit2D &hit, 
		  vector<const GeomDetUnit*> &gd, vector<GlobalPoint> &globHit,
		  unsigned int &hitsAtTecPlus); 
//   void fillVectors(TkFittedLasBeam &beam);

  // need static functions to be used in fitter(..);
  // all parameters used therein have to be static, as well (see below)
  static double tecPlusFunction(double *x, double *par);
  static double tecMinusFunction(double *x, double *par);
  static double atFunction(double *x, double *par);

  void fitter(TkFittedLasBeam &beam, AlgebraicSymMatrix &covMatrix, 
	      unsigned int &hitsAtTecPlus, unsigned int &nFitParams,
	      std::vector<double> &hitPhi, std::vector<double> &hitPhiError, std::vector<double> &hitZprimeError,
	      double &zMin, double &zMax, double &bsAngleParam,
	      double &offset, double &offsetError, double &slope, double &slopeError);

  void trackPhi(TkFittedLasBeam &beam, unsigned int &hit,
		double &trackPhi, double &trackPhiRef,
		double &offset, double &slope, double &bsAngleParam, 
		std::vector<GlobalPoint> &globHit);

  void globalTrackPoint(TkFittedLasBeam &beam, 
			unsigned int &hit, unsigned int &hitsAtTecPlus,
			double &trackPhi, double &trackPhiRef, 
			std::vector<GlobalPoint> &globHit, std::vector<GlobalPoint> &globPtrack,
			GlobalPoint &globPref, std::vector<double> &hitPhiError);

  void buildTrajectory(TkFittedLasBeam &beam, unsigned int &hit,
		       vector<const GeomDetUnit*> &gd, std::vector<GlobalPoint> &globPtrack,
		       vector<TrajectoryStateOnSurface> &tsosLas, GlobalPoint &globPref);

  bool fitBeam(TkFittedLasBeam &beam, AlgebraicSymMatrix &covMatrix, 
	       unsigned int &hitsAtTecPlus, unsigned int &nFitParams,
	       double &offset, double &slope, vector<GlobalPoint> &globPtrack, 
	       double &bsAngleParam, double &chi2);

  // ----------member data ---------------------------
  const edm::InputTag src_;

  // static parameters used in static parametrization functions
  static vector<double> gHitZprime;
  static vector<double> gBarrelModuleRadius;
  static vector<double> gBarrelModuleOffset;
  static double gBeamR;
  static double gBeamZ0;
  static double gBeamSplitterZprime;
  static unsigned int gHitsAtTecMinus;
  static double gBSparam;
  static bool gFitBeamSplitters;

  // fit parameters
//   static double gAtMinusParam;
//   static double gAtMinusParamError;
//   static double gAtPlusParam;
//   static double gAtPlusParamError;
//   static double gPhiAtMinusParam;
//   static double gPhiAtMinusParamError;
//   static double gThetaAtMinusParam;
//   static double gThetaAtMinusParamError;
//   static double gPhiAtPlusParam;
//   static double gPhiAtPlusParamError;
//   static double gThetaAtPlusParam;
//   static double gThetaAtPlusParamError;

  // histograms
  TH1F *h_bsAngle, *h_hitX, *h_hitXTecPlus, *h_hitXTecMinus,
    *h_hitXAt, *h_chi2, *h_chi2ndof, *h_pull, *h_res, 
    *h_resTecPlus, *h_resTecMinus, *h_resAt;
  TH2F *h_bsAngleVsBeam, *h_hitXvsZTecPlus, *h_hitXvsZTecMinus,
    *h_hitXvsZAt, *h_resVsZTecPlus, *h_resVsZTecMinus, *h_resVsZAt,
    *h_resVsHitTecPlus, *h_resVsHitTecMinus, *h_resVsHitAt;
}; 

//
// constants, enums and typedefs
//


//
// static data member definitions
//

// static parameters used in parametrization functions
vector<double> TkLasBeamFitter::gHitZprime;
vector<double> TkLasBeamFitter::gBarrelModuleRadius;
vector<double> TkLasBeamFitter::gBarrelModuleOffset;
double TkLasBeamFitter::gBeamR = 0.0;
double TkLasBeamFitter::gBeamZ0 = 0.0; 
double TkLasBeamFitter::gBeamSplitterZprime = 0.0;
unsigned int TkLasBeamFitter::gHitsAtTecMinus = 0;
double TkLasBeamFitter::gBSparam = 0.0;
bool TkLasBeamFitter::gFitBeamSplitters = 0;

// fit parameters
// double TkLasBeamFitter::gAtMinusParam = 0.0;
// double TkLasBeamFitter::gAtMinusParamError = 0.0;
// double TkLasBeamFitter::gAtPlusParam = 0.0;
// double TkLasBeamFitter::gAtPlusParamError = 0.0;
// double TkLasBeamFitter::gPhiAtMinusParam = 0.0;
// double TkLasBeamFitter::gPhiAtMinusParamError = 0.0;
// double TkLasBeamFitter::gThetaAtMinusParam = 0.0;
// double TkLasBeamFitter::gThetaAtMinusParamError = 0.0;
// double TkLasBeamFitter::gPhiAtPlusParam = 0.0;
// double TkLasBeamFitter::gPhiAtPlusParamError = 0.0;
// double TkLasBeamFitter::gThetaAtPlusParam = 0.0;
// double TkLasBeamFitter::gThetaAtPlusParamError = 0.0;

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
// }
// // FIXME!
// // Indeed, that should be in endRun(..) - as soon as AlignmentProducer can call
// // the algorithm's endRun correctly!
//
//
// void TkLasBeamFitter::beginRun(edm::Run &run, const edm::EventSetup &setup)
// {

  // book histograms
  edm::Service<TFileService> fs;
  h_hitX = fs->make<TH1F>("hitX","local x of LAS hits;local x [cm];N",100,-0.5,0.5);
  h_hitXTecPlus = fs->make<TH1F>("hitXTecPlus","local x of LAS hits in TECplus;local x [cm];N",100,-0.5,0.5);
  h_hitXTecMinus = fs->make<TH1F>("hitXTecMinus","local x of LAS hits in TECminus;local x [cm];N",100,-0.5,0.5);
  h_hitXAt = fs->make<TH1F>("hitXAt","local x of LAS hits in ATs;local x [cm];N",100,-2.5,2.5);
  h_hitXvsZTecPlus = fs->make<TH2F>("hitXvsZTecPlus","local x vs z in TECplus;z [cm];local x [cm]",80,120,280,100,-0.5,0.5);
  h_hitXvsZTecMinus = fs->make<TH2F>("hitXvsZTecMinus","local x vs z in TECMinus;z [cm];local x [cm]",80,-280,-120,100,-0.5,0.5);
  h_hitXvsZAt = fs->make<TH2F>("hitXvsZAt","local x vs z in ATs;z [cm];local x [cm]",200,-200,200,100,-0.5,0.5);
  h_chi2 = fs->make<TH1F>("chi2","#chi^{2};#chi^{2};N",100,0,20000);
  h_chi2ndof = fs->make<TH1F>("chi2ndof","#chi^{2} per degree of freedom;#chi^{2}/N_{dof};N",100,0,3000);
  h_pull = fs->make<TH1F>("pull","pulls of #phi residuals;pull;N",50,-10,10);
  h_res = fs->make<TH1F>("res","#phi residuals;#phi_{track} - #phi_{hit} [rad];N",60,-0.0015,0.0015);
  h_resTecPlus = fs->make<TH1F>("resTecPlus","#phi residuals in TECplus;#phi_{track} - #phi_{hit} [rad];N",30,-0.0015,0.0015);
  h_resTecMinus = fs->make<TH1F>("resTecMinus","#phi residuals in TECminus;#phi_{track} - #phi_{hit} [rad];N",60,-0.0015,0.0015);
  h_resAt = fs->make<TH1F>("resAt","#phi residuals in ATs;#phi_{track} - #phi_{hit} [rad];N",30,-0.0015,0.0015);
  h_resVsZTecPlus = fs->make<TH2F>("resVsZTecPlus","phi residuals vs. z in TECplus;z [cm];#phi_{track} - #phi_{hit} [rad]",
			 80,120,280,100,-0.0015,0.0015);
  h_resVsZTecMinus = fs->make<TH2F>("resVsZTecMinus","phi residuals vs. z in TECminus;z [cm];#phi_{track} - #phi_{hit} [rad]",
			  80,-280,-120,100,-0.0015,0.0015);
  h_resVsZAt = fs->make<TH2F>("resVsZAt","#phi residuals vs. z in ATs;N;#phi_{track} - #phi_{hit} [rad]",
		    200,-200,200,100,-0.0015,0.0015);
  h_resVsHitTecPlus = fs->make<TH2F>("resVsHitTecPlus","#phi residuals vs. hits in TECplus;hit no.;#phi_{track} - #phi_{hit} [rad]",
			   144,0,144,100,-0.0015,0.0015);
  h_resVsHitTecMinus = fs->make<TH2F>("resVsHitTecMinus","#phi residuals vs. hits in TECminus;hit no.;#phi_{track} - #phi_{hit} [rad]",
			    144,0,144,100,-0.0015,0.0015);
  h_resVsHitAt = fs->make<TH2F>("resVsHitAt","#phi residuals vs. hits in ATs;hit no.;#phi_{track} - #phi_{hit} [rad]",
		      176,0,176,100,-0.0015,0.0015);
  h_bsAngle = fs->make<TH1F>("bsAngle","fitted beam splitter angle;BS angle [rad];N",40,-0.004,0.004);
  h_bsAngleVsBeam = fs->make<TH2F>("bsAngleVsBeam","fitted beam splitter angle per beam;Beam no.;BS angle [rad]",
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

  // hack for fixed BSparams
  double bsParams[34] = {-0.0017265,-0.00115254,-0.00107988,-0.000203643,-0.000561458,
			 0.0010246,0.000399215,0.00216394,0.000943104,0.00186308,
			 -0.00140239,-0.00477247,-0.00230141,0.000229482,-0.00317751,
			 0.00151446,0.00337618,-0.00473474,-0.000555934,0.00026294,
			 -0.00214642,-0.000314923,-0.000552433,0.000282853,-0.00132737,-0.00163106,
			 // ATs
			 0.,-0.000267358,-0.0029504,0.,
			 -0.00180776,0.,-0.0016999,-0.00217921};

//   double bsParams[40] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
// 			 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  // beam counter
  unsigned int beamNo = 0;
  // fit BS? If false, values from bsParams are taken
  gFitBeamSplitters = true;
//   if(gFitBeamSplitters) cout << "Fitting BS!" << endl;
//   else cout << "BS fixed, not fitted!" << endl;

  // loop over LAS beams
  for(TkLasBeamCollection::const_iterator iBeam = laserBeams->begin(), iEnd = laserBeams->end();
       iBeam != iEnd; ++iBeam){

    TkFittedLasBeam beam(*iBeam);
    vector<TrajectoryStateOnSurface> tsosLas;

    // set BS param for fit
    gBSparam = bsParams[beamNo];
//     cout << "Beam Number " << beamNo << ", BSparam = " << gBSparam << endl;

    // call main function; all other functions are called inside getLasBeams(..)
    this->getLasBeams(beam, tsosLas);
    
    // fill output products
    fittedBeams->push_back(beam);
//     cout << "No. of TSOS in beam: " << tsosLas.size() << endl;
    tsosesVec->push_back(tsosLas);

//     if(!this->fitBeam(fittedBeams->back(), tsosesVec->back())){
//       edm::LogError("BadFit") 
// 	 << "Problems fitting TkLasBeam, id " << fittedBeams->back().getBeamId() << ".";
//       fittedBeams->pop_back(); // remove last entry added just before
//       tsosesVec->pop_back();   // dito
//     }

    beamNo++;
  }
  
  // finally, put fitted beams and TSOS vectors into run
  run.put(fittedBeams);
  run.put(tsosesVec);
}

// methods for las data processing

// -------------- loop over beams, call functions ----------------------------
void TkLasBeamFitter::getLasBeams(TkFittedLasBeam &beam, vector<TrajectoryStateOnSurface> &tsosLas)
{
//   cout << "---------------------------------------" << endl;
//   cout << "beam id: " << beam.getBeamId() // << " isTec: " << (beam.isTecInternal() ? "Y" : "N") 
//        << " isTec+: " << (beam.isTecInternal(1) ? "Y" : "N") << " isTec-: " << (beam.isTecInternal(-1) ? "Y" : "N")
//        << " isAt: " << (beam.isAlignmentTube() ? "Y" : "N") << " isR6: " << (beam.isRing6() ? "Y" : "N")
//        << endl;
  
  gHitsAtTecMinus = 0;
  gHitZprime.clear();
  gBarrelModuleRadius.clear();
  gBarrelModuleOffset.clear();

  // set right beam radius
  gBeamR = beam.isRing6() ? 84.0 : 56.4;
//   cout << "nominal beam radius = " << gBeamR << endl;
  
  vector<const GeomDetUnit*> gd;
  vector<GlobalPoint> globHit;
  unsigned int hitsAtTecPlus = 0;
  double sumZ = 0.0;

  // loop over hits
  for( TkLasBeam::const_iterator iHit = beam.begin(); iHit < beam.end(); ++iHit ){
    // iHit is a SiStripLaserRecHit2D

//     cout << "Strip hit (local x): " << hit.localPosition().x() 
// 	 << " +- " << hit.localPositionError().xx() << endl;

    SiStripLaserRecHit2D hit(*iHit);
    this->getLasHits(beam, hit, gd, globHit, hitsAtTecPlus);
    sumZ += globHit.back().z();

    // fill histos
    h_hitX->Fill(hit.localPosition().x());
    // TECplus
    if(beam.isTecInternal(1)){
      h_hitXTecPlus->Fill(hit.localPosition().x());
      h_hitXvsZTecPlus->Fill(globHit.back().z(),hit.localPosition().x());
    }
    // TECminus
    else if(beam.isTecInternal(-1)){
      h_hitXTecMinus->Fill(hit.localPosition().x());
      h_hitXvsZTecMinus->Fill(globHit.back().z(),hit.localPosition().x());
    }
    // ATs
    else{
      h_hitXAt->Fill(hit.localPosition().x());
      h_hitXvsZAt->Fill(globHit.back().z(),hit.localPosition().x());
    }  
  }

  gBeamZ0 = sumZ / globHit.size();
  double zMin = 0.0;
  double zMax = 0.0;
  // TECplus
  if(beam.isTecInternal(1)){
    gBeamSplitterZprime = 205.75 - gBeamZ0;
    zMin = 120.0 - gBeamZ0;
    zMax = 280.0 - gBeamZ0;
  }
  // TECminus
  else if(beam.isTecInternal(-1)){
    gBeamSplitterZprime = -205.75 - gBeamZ0;
    zMin = -280.0 - gBeamZ0;
    zMax = -120.0 - gBeamZ0;
  }
  // AT
  else{
    gBeamSplitterZprime = 112.3 - gBeamZ0;
    zMin = -200.0 - gBeamZ0;
    zMax = 200.0 - gBeamZ0;
  }
//   cout << "z0 = " << gBeamZ0 << ", z_BS' = " << gBeamSplitterZprime 
//        << ", zMin = " << zMin << ", zMax = " << zMax << endl;

  // fill vectors for fitted quantities
  vector<double> hitPhi;
  vector<double> hitPhiError;
  vector<double> hitZprimeError;

  for(unsigned int hit = 0; hit < globHit.size(); ++hit){
    hitPhi.push_back(static_cast<double>(globHit[hit].phi()));
    // localPositionError[hit] or assume 0.003
    hitPhiError.push_back( 0.003 / globHit[hit].perp());
    // no errors on z, fill with zeros
    hitZprimeError.push_back(0.0);
    // barrel-specific values
    if(beam.isAlignmentTube() && abs(globHit[hit].z()) < 112.3){
      gBarrelModuleRadius.push_back(globHit[hit].perp());
      gBarrelModuleOffset.push_back(abs(gBarrelModuleRadius.back() - gBeamR));
      gHitZprime.push_back(globHit[hit].z() - gBeamZ0 - gBarrelModuleOffset.back());
//       cout << "module offset = " << gBarrelModuleOffset[hit-gHitsAtTecMinus] 
// 	   << ", radius = " << gBarrelModuleRadius[hit-gHitsAtTecMinus] 
// 	   << endl;
    }
    // non-barrel z'
    else{
      gHitZprime.push_back(globHit[hit].z() - gBeamZ0);
    }
  }

  // number of fit parameters
  unsigned int tecParams = 3; // maximum number
  unsigned int atParams = 3; // maximum number (currently)
  unsigned int nFitParams = 0;
  if(!gFitBeamSplitters || 
     (gHitsAtTecMinus == 0 && beam.isAlignmentTube() ) ){
    tecParams = tecParams - 1;
    atParams = atParams - 1;
  }
  if(beam.isTecInternal()){
    nFitParams = tecParams;
  }
  else{
    nFitParams = atParams;
  }
  
  double offset = 0.0;
  double offsetError = 0.0;
  double slope = 0.0;
  double slopeError = 0.0;
  double bsAngleParam = 0.0;
  AlgebraicSymMatrix covMatrix;
  if(!gFitBeamSplitters || (beam.isAlignmentTube() && hitsAtTecPlus == 0)){
    covMatrix = AlgebraicSymMatrix(nFitParams, 1);
  }
  else{
    covMatrix = AlgebraicSymMatrix(nFitParams - 1, 1);
  }

//   cout << "used hits: " << gHitZprime.size()
//        << ", nBarrelModules: " << gBarrelModuleRadius.size() 
//        << ", hits in AT TECminus: " << gHitsAtTecMinus
//        << ", hits in AT TECplus: " << hitsAtTecPlus << endl;

  this->fitter(beam, covMatrix, 
	       hitsAtTecPlus, nFitParams, 
	       hitPhi, hitPhiError, hitZprimeError, 
	       zMin, zMax, bsAngleParam,
	       offset, offsetError, slope, slopeError); 
  
  vector<GlobalPoint> globPtrack;
  GlobalPoint globPref;
  double chi2 = 0.0;

  for(unsigned int hit = 0; hit < gHitZprime.size(); ++hit){
    
    double trackPhi;
    // additional phi value for trajectory calculation
    double trackPhiRef;   
    
    this->trackPhi(beam, hit, trackPhi, trackPhiRef,
		   offset, slope, bsAngleParam, globHit);

//     cout //<< "track phi = " << trackPhi 
// 	 << ", hit phi = " << hitPhi[hit] 
// 	 << ", zPrime = " << gHitZprime[hit] << endl;

    this->globalTrackPoint(beam, hit, hitsAtTecPlus, 
			   trackPhi, trackPhiRef, 
			   globHit, globPtrack, globPref, 
			   hitPhiError);

//   cout << "hit (global): " << globHit[hit] 
//        << " r = " << globHit[hit].perp() << " phi = " << globHit[hit].phi() 
//        << endl;
//        << "  fitted hit: " << globPtrack[hit] 
//        << " r = " << globPtrack[hit].perp() << ", phi = " << globPtrack[hit].phi() 
//        << endl;
//        << "    reference point: " << globPref << " r = " << globPref.perp() 
//        << ", phi = " << globPref.phi() << endl;
  
    // calculate residuals = pred - hit (in global phi)
    const double phiResidual = globPtrack[hit].phi() - globHit[hit].phi();
    // pull calculation (FIX!!!)
    const double phiResidualPull = phiResidual / hitPhiError[hit];
    //       sqrt(hitPhiError[hit]*hitPhiError[hit] + 
    // 	   (offsetError*offsetError + globPtrack[hit].z()*globPtrack[hit].z() * slopeError*slopeError));
    //   cout << "phi residual = " << phiResidual << endl
    //        << "phi residual * r = " << phiResidual*globPtrack[hit].perp() << endl;
    // calculate chi2
    chi2 += phiResidual*phiResidual / (hitPhiError[hit]*hitPhiError[hit]);
    
    // fill histos
    h_res->Fill(phiResidual);
    // TECplus
    if(beam.isTecInternal(1)){
      h_pull->Fill(phiResidualPull);
      h_resTecPlus->Fill(phiResidual);
      h_resVsZTecPlus->Fill(globPtrack[hit].z(), phiResidual);
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
      h_resVsZTecMinus->Fill(globPtrack[hit].z(), phiResidual);
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
      h_resVsZAt->Fill(globPtrack[hit].z(), phiResidual);
      h_resVsHitAt->Fill(hit+(beam.getBeamId()-200)/10*22, phiResidual);
    }

    this->buildTrajectory(beam, hit, gd, globPtrack, tsosLas, globPref);

//     cout << "TSOS local position " << tsosLas[hit].localPosition().x() << endl
// 	 << "LAS hit local position " << beam.getData()[hit].localPosition().x() << endl
// 	 << "local residual = " << beam.getData()[hit].localPosition().x() - tsosLas[hit].localPosition().x() << endl
// 	 << "local res / r = " 
// 	 << (beam.getData()[hit].localPosition().x() - tsosLas[hit].localPosition().x()) / globHit[hit].perp() << endl;
//     cout << "hit (beam.getData): " << GlobalPoint(gd[hit]->toGlobal(beam.getData()[hit].localPosition())) << endl;
// 	 << "  fitted hit from TSOS: " << GlobalPoint(gd[hit]->toGlobal(tsosLas[hit].localPosition())) << endl;
  }

  cout << "chi^2 = " << chi2 << ", chi^2/ndof = " << chi2/(gHitZprime.size() - nFitParams) << endl;
  this->fitBeam(beam, covMatrix, hitsAtTecPlus, nFitParams,
		offset, slope, globPtrack, bsAngleParam, chi2);
    
  // fill histos
  
  // include slope, offset, covariance plots here
  h_chi2->Fill(chi2);
  h_chi2ndof->Fill(chi2/(gHitZprime.size() - nFitParams));
  if(bsAngleParam != 0.0){
    h_bsAngle->Fill(2.0*atan(0.5*bsAngleParam));
    h_bsAngleVsBeam->Fill(beam.getBeamId(), 2.0*atan(0.5*bsAngleParam));
  }
}


// --------- get hits, convert to global coordinates ---------------------------
void TkLasBeamFitter::getLasHits(TkFittedLasBeam &beam, SiStripLaserRecHit2D &hit, 
				 vector<const GeomDetUnit*> &gd, vector<GlobalPoint> &globHit,
				 unsigned int &hitsAtTecPlus)
{ 
  // get global position of LAS hits
  gd.push_back(geometry->idToDetUnit(hit.getDetId()));
  // Global position gives wrong radius!
  GlobalPoint globPtemp(gd.back()->toGlobal(hit.localPosition()));  
  // Vector globHit is filled with nominal beam radii as radial position.
  // TECs
  if(beam.isTecInternal()){
    globHit.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, globPtemp.phi(), globPtemp.z())));
  }
  // ATs
  else{
    // TECs
    if(abs(globPtemp.z()) > 112.3){
      if(globPtemp.z() < 112.3) gHitsAtTecMinus++ ;
      else hitsAtTecPlus++ ;
      globHit.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, globPtemp.phi(), globPtemp.z())));
      }
    // Barrel
    else{
      globHit.push_back(globPtemp);
    }
  }
}

// void TkLasBeamFitter::setZvalues(TkFittedLasBeam &beam)
// {
// }


// void TkLasBeamFitter::fillVectors(TkFittedLasBeam &beam)
// {
// }


// ------------ parametrization functions for las beam fits ------------
double TkLasBeamFitter::tecPlusFunction(double *x, double *par)
{
  double z = x[0]; // 'primed'? -> yes!!!

  if(z < gBeamSplitterZprime){
    return par[0] + par[1] * z;
  } 
  else{
    if(gFitBeamSplitters){
      // par[2] = 2*tan(BeamSplitterAngle/2.0)
      return par[0] + par[1] * z - par[2] * (z - gBeamSplitterZprime)/gBeamR; 
    }
    else{
      return par[0] + par[1] * z - gBSparam * (z - gBeamSplitterZprime)/gBeamR; 
    }
  }
}


double TkLasBeamFitter::tecMinusFunction(double *x, double *par)
{
  double z = x[0]; // 'primed'? -> yes!!!

  if(z > gBeamSplitterZprime){
    return par[0] + par[1] * z;
  } 
  else{
    if(gFitBeamSplitters){
      // par[2] = 2*tan(BeamSplitterAngle/2.0)
      return par[0] + par[1] * z + par[2] * (z - gBeamSplitterZprime)/gBeamR; 
    }
    else{
      return par[0] + par[1] * z + gBSparam * (z - gBeamSplitterZprime)/gBeamR; 
    }
  }
}
 
double TkLasBeamFitter::atFunction(double *x, double *par)
{
  double z = x[0]; // 'primed'? -> yes!!!
  // TECminus
  if(z < -gBeamSplitterZprime - 2.0*gBeamZ0){
    return par[0] + par[1] * z;
  }
  // Barrel
  else if(-gBeamSplitterZprime - 2.0*gBeamZ0 < z && z < gBeamSplitterZprime){
    // z includes module offset from main beam axis
    return par[0] + par[1] * z;
    // later, include tilts/rotations of ATs
    // par[3] = tan(phiAtMinus), par[5] = tan(2.0*thetaAtMinus)
    // par[4] = tan(phiAtPlus), par[6] = tan(2.0*thetaAtPlus)
  }
  // TECplus
  else{
    if(gFitBeamSplitters){
      // par[2] = 2*tan(BeamSplitterAngle/2.0)
      return par[0] + par[1] * z - par[2] * (z - gBeamSplitterZprime)/gBeamR;
    }
    else{
      return par[0] + par[1] * z - gBSparam * (z - gBeamSplitterZprime)/gBeamR;
    }
  }
}


// ------------ perform fit of beams ------------------------------------
void TkLasBeamFitter::fitter(TkFittedLasBeam &beam, AlgebraicSymMatrix &covMatrix, 
			     unsigned int &hitsAtTecPlus, unsigned int &nFitParams,
			     vector<double> &hitPhi, vector<double> &hitPhiError, vector<double> &hitZprimeError,
			     double &zMin, double &zMax, double &bsAngleParam, 
			     double &offset, double &offsetError, double &slope, double &slopeError)
{
  TGraphErrors *lasData = new TGraphErrors(gHitZprime.size(), 
					   &(gHitZprime[0]), &(hitPhi[0]), 
					   &(hitZprimeError[0]), &(hitPhiError[0]));
  
  // do fit (R = entire range)
  if(beam.isTecInternal(1)){
    TF1 tecPlus("tecPlus", tecPlusFunction, zMin, zMax, nFitParams );
    lasData->Fit("tecPlus", "R"); // "R", "RV" or "RQ"
  }
  else if(beam.isTecInternal(-1)){
    TF1 tecMinus("tecMinus", tecMinusFunction, zMin, zMax, nFitParams );
    lasData->Fit("tecMinus", "R");
  }
  else{
    TF1 at("at", atFunction, zMin, zMax, nFitParams ); 
    lasData->Fit("at","R");
  }
  
  double bsAngleParamError = 0.0;
  // get values and errors for offset and slope
  gMinuit->GetParameter(0, offset, offsetError);
  gMinuit->GetParameter(1, slope, slopeError);
  // ...and BS angle if applicable
  if(gFitBeamSplitters || (beam.isAlignmentTube() && hitsAtTecPlus > 0) ) {
    gMinuit->GetParameter(2, bsAngleParam, bsAngleParamError);
  }
  else{
    bsAngleParam = gBSparam;
  }

  // additional AT parameters (not working, yet)
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

  // fill covariance matrix
  vector<double> vec(covMatrix.num_col()*covMatrix.num_col());
  gMinuit->mnemat(&vec[0], covMatrix.num_col());
  // fill covariance matrix
  for(int col = 0; col < covMatrix.num_col(); col++){
    for(int row = 0; row < covMatrix.num_col(); row++){
      covMatrix[col][row] = vec[row + covMatrix.num_col()*col];
    }
  }
  // compute correlation between parameters
//   double corr01 = covMatrix[1][0]/(offsetError*slopeError);

  // display fit results
//   cout << "number of parameters: " << nFitParams << endl
//        << "a = " << offset << ", aError = " << offsetError << endl 
//        << "b = " << slope  << ", bError = " << slopeError << endl;
//   cout << "c = " << bsAngleParam << ", cError = " << bsAngleParamError << endl 
//        << "BSangle = " << 2.0*atan(0.5*bsAngleParam) << ", BSangleError = " 
//        << bsAngleParamError/(1+bsAngleParam*bsAngleParam/4.0) << endl;
//   cout << "correlation between offset and slope: " << corr01 << endl;

  delete lasData;
}


// -------------- calculate track phi value ----------------------------------
void TkLasBeamFitter::trackPhi(TkFittedLasBeam &beam, unsigned int &hit, 
			       double &trackPhi, double &trackPhiRef,
			       double &offset, double &slope, double &bsAngleParam,
			       vector<GlobalPoint> &globHit)
{
  // TECplus
  if(beam.isTecInternal(1)){
    if(gHitZprime[hit] < gBeamSplitterZprime){
      trackPhi = offset + slope * gHitZprime[hit];
      trackPhiRef = offset + slope * (gHitZprime[hit] + 1.0);
    }
    else{
      trackPhi = offset + slope * gHitZprime[hit] 
	- bsAngleParam * (gHitZprime[hit] - gBeamSplitterZprime)/gBeamR;
      trackPhiRef = offset + slope * (gHitZprime[hit] + 1.0)
	- bsAngleParam * ((gHitZprime[hit] + 1.0) - gBeamSplitterZprime)/gBeamR;
    }
  }
  // TECminus
  else if(beam.isTecInternal(-1)){
    if(gHitZprime[hit] > gBeamSplitterZprime){
      trackPhi = offset + slope * gHitZprime[hit];
      trackPhiRef = offset + slope * (gHitZprime[hit] + 1.0);
    }
    else{
      trackPhi = offset + slope * gHitZprime[hit] 
	+ bsAngleParam * (gHitZprime[hit] - gBeamSplitterZprime)/gBeamR;
      trackPhiRef = offset + slope * (gHitZprime[hit] + 1.0)
	+ bsAngleParam * ((gHitZprime[hit] + 1.0) - gBeamSplitterZprime)/gBeamR;
    }
  }
  // ATs
  else{
    // TECminus
    if(gHitZprime[hit] < -gBeamSplitterZprime - 2.0*gBeamZ0){
      trackPhi = offset + slope * gHitZprime[hit];
      trackPhiRef = offset + slope * (gHitZprime[hit] + 1.0);
    }
    // Barrel
    else if(-gBeamSplitterZprime - 2.0*gBeamZ0 < gHitZprime[hit] && gHitZprime[hit] < gBeamSplitterZprime){
      trackPhi = offset + slope * gHitZprime[hit];
      trackPhiRef = offset + slope * (gHitZprime[hit] + gBarrelModuleOffset[hit-gHitsAtTecMinus]);
    }
    // TECplus
    else{
      trackPhi = offset + slope * gHitZprime[hit] 
	- bsAngleParam * (gHitZprime[hit] - gBeamSplitterZprime)/gBeamR;
      trackPhiRef = offset + slope * (gHitZprime[hit] + 1.0)
	- bsAngleParam * ((gHitZprime[hit] + 1.0) - gBeamSplitterZprime)/gBeamR;
    }
  }
}


// -------------- calculate global track points, hit residuals, chi2 ----------------------------------
void TkLasBeamFitter::globalTrackPoint(TkFittedLasBeam &beam, 
				       unsigned int &hit, unsigned int &hitsAtTecPlus,
				       double &trackPhi, double &trackPhiRef, 
				       vector<GlobalPoint> &globHit, vector<GlobalPoint> &globPtrack,
				       GlobalPoint &globPref, vector<double> &hitPhiError)
{
  // TECs
  if(beam.isTecInternal(0)){
    globPtrack.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhi, globHit[hit].z())));
    globPref = GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhiRef, globHit[hit].z() + 1.0));
  }
  // ATs
  else{
    // TECminus
    if(hit < gHitsAtTecMinus){ // gHitZprime[hit] < -gBeamSplitterZprime - 2.0*gBeamZ0
      globPtrack.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhi, globHit[hit].z())));
      globPref = GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhiRef, globHit[hit].z() + 1.0));
    }
    // TECplus
    else if(hit > gHitZprime.size() - hitsAtTecPlus - 1){ // gHitZprime[hit] > gBeamSplitterZprime
      globPtrack.push_back(GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhi, globHit[hit].z())));
      globPref = GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhiRef, globHit[hit].z() + 1.0));
    }
    // Barrel
    else{
      globPtrack.push_back(GlobalPoint(GlobalPoint::Cylindrical(globHit[hit].perp(), trackPhi, globHit[hit].z())));
      globPref = GlobalPoint(GlobalPoint::Cylindrical(gBeamR, trackPhiRef, globHit[hit].z()));
    }
  }
}


// ----------- create TrajectoryStateOnSurface for each track hit ----------------------------------------------
void TkLasBeamFitter::buildTrajectory(TkFittedLasBeam &beam, unsigned int &hit, 
				      vector<const GeomDetUnit*> &gd, vector<GlobalPoint> &globPtrack,
				      vector<TrajectoryStateOnSurface> &tsosLas, GlobalPoint &globPref)
{
  const MagneticField* magneticField = fieldHandle.product();
  GlobalVector trajectoryState;
  
  // TECplus
  if(beam.isTecInternal(1)){
    trajectoryState = GlobalVector(globPref-globPtrack[hit]);
  }
  // TECminus
  else if(beam.isTecInternal(-1)){
    trajectoryState = GlobalVector(globPtrack[hit]-globPref);
  }
  // ATs
  else{
    // TECminus
    if(gHitZprime[hit] < -gBeamSplitterZprime - 2.0*gBeamZ0){
      trajectoryState = GlobalVector(globPtrack[hit]-globPref);
    }
    // TECplus
    else if(gHitZprime[hit] > gBeamSplitterZprime){
      trajectoryState = GlobalVector(globPref-globPtrack[hit]);
    }
    // Barrel
    else{
      trajectoryState = GlobalVector(globPtrack[hit]-globPref);
    }
  }	
//   cout << "trajectory: " << trajectoryState << endl;			
  const FreeTrajectoryState ftsLas = FreeTrajectoryState(globPtrack[hit],trajectoryState,0,magneticField);
  tsosLas.push_back(TrajectoryStateOnSurface(ftsLas,gd[hit]->surface(),
					      SurfaceSideDefinition::beforeSurface));
}


//---------------------- set beam parameters for fittedBeams ---------------------------------
bool TkLasBeamFitter::fitBeam(TkFittedLasBeam &beam, AlgebraicSymMatrix &covMatrix, 
			      unsigned int &hitsAtTecPlus, unsigned int &nFitParams,
			      double &offset, double &slope, vector<GlobalPoint> &globPtrack,
			      double &bsAngleParam, double &chi2)
{
  // set beam parameters for beam output
  unsigned int paramType = 0;
  if(!gFitBeamSplitters) paramType = 1;
  if(beam.isAlignmentTube() && hitsAtTecPlus == 0) paramType = 0;
  const unsigned int nPedeParams = nFitParams + paramType;
//   cout << "number of Pede parameters: " << nPedeParams << endl;

  std::vector<TkFittedLasBeam::Scalar> params(nPedeParams);
  params[0] = offset;
  params[1] = slope;
  // no BS parameter for AT beams without TECplus hits
  if(beam.isTecInternal() || hitsAtTecPlus > 0) params[2] = bsAngleParam;

  AlgebraicMatrix derivatives(gHitZprime.size(), nPedeParams);
  // fill derivatives matrix with local track derivatives
  for(unsigned int hit = 0; hit < gHitZprime.size(); ++hit){
    
    // d(delta phi)/d(offset) is identical for every hit
    derivatives[hit][0] = 1.0;

    // d(delta phi)/d(slope) and d(delta phi)/d(bsAngleParam) depend on parametrizations
    // TECplus
    if(beam.isTecInternal(1)){
      derivatives[hit][1] = globPtrack[hit].z();
      if(gHitZprime[hit] < gBeamSplitterZprime){
	derivatives[hit][2] = 0.0;
      }
      else{
	derivatives[hit][2] = - (globPtrack[hit].z() - gBeamSplitterZprime) / gBeamR;
      }
    }
    // TECminus
    else if(beam.isTecInternal(-1)){
      derivatives[hit][1] = globPtrack[hit].z();
      if(gHitZprime[hit] > gBeamSplitterZprime){
	derivatives[hit][2] = 0.0;
      }
      else{
	derivatives[hit][2] = (globPtrack[hit].z() - gBeamSplitterZprime) / gBeamR;
      }
    }
    // ATs
    else{
      // TECminus
      if(gHitZprime[hit] < -gBeamSplitterZprime - 2.0*gBeamZ0){
	derivatives[hit][1] = globPtrack[hit].z();
	if(hitsAtTecPlus > 0){
	  derivatives[hit][2] = 0.0;
	}
      }
      // TECplus
      else if(gHitZprime[hit] > gBeamSplitterZprime){
	derivatives[hit][1] = globPtrack[hit].z();
	if(hitsAtTecPlus > 0){
	  derivatives[hit][2] = - (globPtrack[hit].z() - gBeamSplitterZprime) / gBeamR;
	}
      }
      // Barrel
      else{
	derivatives[hit][1] = globPtrack[hit].z() - gBarrelModuleOffset[hit-gHitsAtTecMinus];
	if(hitsAtTecPlus > 0){
	  derivatives[hit][2] = 0.0;
	}
      }
    }
  }

  unsigned int firstFixedParam = covMatrix.num_col(); // FIXME!
//   unsigned int firstFixedParam = nPedeParams - 1;
//   if(beam.isAlignmentTube() && hitsAtTecPlus == 0) firstFixedParam = nPedeParams;
//   cout << "first fixed parameter: " << firstFixedParam << endl;
  // set fit results
  beam.setParameters(paramType, params, covMatrix, derivatives, firstFixedParam, chi2);

  return true; // return false in case of problems
}


//---------------------------------------------------------------------------------------
//define this as a plug-in
DEFINE_FWK_MODULE(TkLasBeamFitter);
