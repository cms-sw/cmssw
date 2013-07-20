/**\class TkLasBeamFitter TkLasBeamFitter.cc Alignment/LaserAlignment/plugins/TkLasBeamFitter.cc

  Original Authors:  Gero Flucke/Kolja Kaschube
           Created:  Wed May  6 08:43:02 CEST 2009
           $Id: TkLasBeamFitter.cc,v 1.13 2013/05/17 18:12:18 chrjones Exp $

 Description: Fitting LAS beams with track model and providing TrajectoryStateOnSurface for hits.

 Implementation:
    - TkLasBeamCollection read from edm::Run
    - all done in endRun(..) to allow a correct sequence with 
      production of TkLasBeamCollection in LaserAlignment::endRun(..)
*/



// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

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

class TkLasBeamFitter : public edm::one::EDProducer<edm::EndRunProducer> {
public:
  explicit TkLasBeamFitter(const edm::ParameterSet &config);
  ~TkLasBeamFitter();
  
  //virtual void beginJob(const edm::EventSetup& /*access deprecated*/) {}
  virtual void produce(edm::Event &event, const edm::EventSetup &setup) override;
  // virtual void beginRun(edm::Run &run, const edm::EventSetup &setup);
  virtual void endRunProduce(edm::Run &run, const edm::EventSetup &setup) override;
  //virtual void endJob() {}

private:
  /// Fit 'beam' using info from its base class TkLasBeam and set its parameters.
  /// Also fill 'tsoses' with TSOS for each LAS hit. 
  void getLasBeams(TkFittedLasBeam &beam,vector<TrajectoryStateOnSurface> &tsosLas);
  void getLasHits(TkFittedLasBeam &beam, const SiStripLaserRecHit2D &hit, 
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
	      double &offset, double &offsetError, double &slope, double &slopeError,
	      double &phiAtMinusParam, double &phiAtPlusParam,
	      double &atThetaSplitParam);

  void trackPhi(TkFittedLasBeam &beam, unsigned int &hit,
		double &trackPhi, double &trackPhiRef,
		double &offset, double &slope, double &bsAngleParam, 
		double &phiAtMinusParam, double &phiAtPlusParam,
		double &atThetaSplitParam, std::vector<GlobalPoint> &globHit);

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
  bool fitBeamSplitters_;
  unsigned int nAtParameters_;

  edm::Service<TFileService> fs;

  // static parameters used in static parametrization functions
  static vector<double> gHitZprime;
  static vector<double> gBarrelModuleRadius;
  static vector<double> gBarrelModuleOffset;
  static float gTIBparam;
  static float gTOBparam;
  static double gBeamR;
  static double gBeamZ0;
  static double gBeamSplitterZprime;
  static unsigned int gHitsAtTecMinus;
  static double gBSparam;
  static bool gFitBeamSplitters;
  static bool gIsInnerBarrel;

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
float TkLasBeamFitter::gTIBparam = 0.097614; // = abs(r_offset/r_module) (nominal!)
float TkLasBeamFitter::gTOBparam = 0.034949; // = abs(r_offset/r_module) (nominal!)
double TkLasBeamFitter::gBeamR = 0.0;
double TkLasBeamFitter::gBeamZ0 = 0.0; 
double TkLasBeamFitter::gBeamSplitterZprime = 0.0;
unsigned int TkLasBeamFitter::gHitsAtTecMinus = 0;
double TkLasBeamFitter::gBSparam = 0.0;
bool TkLasBeamFitter::gFitBeamSplitters = 0;
bool TkLasBeamFitter::gIsInnerBarrel = 0;

// handles
Handle<TkLasBeamCollection> laserBeams;
ESHandle<MagneticField> fieldHandle;
ESHandle<TrackerGeometry> geometry;

//
// constructors and destructor
//
TkLasBeamFitter::TkLasBeamFitter(const edm::ParameterSet &iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src")),
  fitBeamSplitters_(iConfig.getParameter<bool>("fitBeamSplitters")),
  nAtParameters_(iConfig.getParameter<unsigned int>("numberOfFittedAtParameters")),
  h_bsAngle(0), h_hitX(0), h_hitXTecPlus(0), h_hitXTecMinus(0),
  h_hitXAt(0), h_chi2(0), h_chi2ndof(0), h_pull(0), h_res(0), 
  h_resTecPlus(0), h_resTecMinus(0), h_resAt(0),
  h_bsAngleVsBeam(0), h_hitXvsZTecPlus(0), h_hitXvsZTecMinus(0),
  h_hitXvsZAt(0), h_resVsZTecPlus(0), h_resVsZTecMinus(0), h_resVsZAt(0),
  h_resVsHitTecPlus(0), h_resVsHitTecMinus(0), h_resVsHitAt(0)
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
void TkLasBeamFitter::endRunProduce(edm::Run &run, const edm::EventSetup &setup)
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
  h_hitX = fs->make<TH1F>("hitX","local x of LAS hits;local x [cm];N",100,-0.5,0.5);
  h_hitXTecPlus = fs->make<TH1F>("hitXTecPlus","local x of LAS hits in TECplus;local x [cm];N",100,-0.5,0.5);
  h_hitXTecMinus = fs->make<TH1F>("hitXTecMinus","local x of LAS hits in TECminus;local x [cm];N",100,-0.5,0.5);
  h_hitXAt = fs->make<TH1F>("hitXAt","local x of LAS hits in ATs;local x [cm];N",100,-2.5,2.5);
  h_hitXvsZTecPlus = fs->make<TH2F>("hitXvsZTecPlus","local x vs z in TECplus;z [cm];local x [cm]",80,120,280,100,-0.5,0.5);
  h_hitXvsZTecMinus = fs->make<TH2F>("hitXvsZTecMinus","local x vs z in TECMinus;z [cm];local x [cm]",80,-280,-120,100,-0.5,0.5);
  h_hitXvsZAt = fs->make<TH2F>("hitXvsZAt","local x vs z in ATs;z [cm];local x [cm]",200,-200,200,100,-0.5,0.5);
  h_chi2 = fs->make<TH1F>("chi2","#chi^{2};#chi^{2};N",100,0,2000);
  h_chi2ndof = fs->make<TH1F>("chi2ndof","#chi^{2} per degree of freedom;#chi^{2}/N_{dof};N",100,0,300);
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

  // hack for fixed BSparams (ugly!)
//   double bsParams[34] = {-0.000266,-0.000956,-0.001205,-0.000018,-0.000759,0.002554,
// 			 0.000465,0.000975,0.001006,0.002027,-0.001263,-0.000763,
// 			 -0.001702,0.000906,-0.002120,0.001594,0.000661,-0.000457,
// 			 -0.000447,0.000347,-0.002266,-0.000446,0.000659,0.000018,
// 			 -0.001630,-0.000324,
// 			 // ATs
// 			 -999.,-0.001709,-0.002091,-999.,
// 			 -0.001640,-999.,-0.002444,-0.002345};

  double bsParams[40] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
			 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  // beam counter
  unsigned int beamNo(0);
  // fit BS? If false, values from bsParams are taken
  gFitBeamSplitters = fitBeamSplitters_;
  if(fitBeamSplitters_) cout << "Fitting BS!" << endl;
  else cout << "BS fixed, not fitted!" << endl;

  // loop over LAS beams
  for(TkLasBeamCollection::const_iterator iBeam = laserBeams->begin(), iEnd = laserBeams->end();
       iBeam != iEnd; ++iBeam){

    TkFittedLasBeam beam(*iBeam);
    vector<TrajectoryStateOnSurface> tsosLas;

    // set BS param for fit
    gBSparam = bsParams[beamNo];

    // call main function; all other functions are called inside getLasBeams(..)
    this->getLasBeams(beam, tsosLas);
    
    // fill output products
    fittedBeams->push_back(beam);
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
  cout << "---------------------------------------" << endl;
  cout << "beam id: " << beam.getBeamId() // << " isTec: " << (beam.isTecInternal() ? "Y" : "N") 
       << " isTec+: " << (beam.isTecInternal(1) ? "Y" : "N") << " isTec-: " << (beam.isTecInternal(-1) ? "Y" : "N")
       << " isAt: " << (beam.isAlignmentTube() ? "Y" : "N") << " isR6: " << (beam.isRing6() ? "Y" : "N")
       << endl;
 
  // reset static variables 
  gHitsAtTecMinus = 0;
  gHitZprime.clear();
  gBarrelModuleRadius.clear();
  gBarrelModuleOffset.clear();

  // set right beam radius
  gBeamR = beam.isRing6() ? 84.0 : 56.4;
  
  vector<const GeomDetUnit*> gd;
  vector<GlobalPoint> globHit;
  unsigned int hitsAtTecPlus(0);
  double sumZ(0.);

  // loop over hits
  for( TkLasBeam::const_iterator iHit = beam.begin(); iHit < beam.end(); ++iHit ){
    // iHit is a SiStripLaserRecHit2D

    const SiStripLaserRecHit2D hit(*iHit);

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
  double zMin(0.), zMax(0.);
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

  // fill vectors for fitted quantities
  vector<double> hitPhi, hitPhiError, hitZprimeError;

  for(unsigned int hit = 0; hit < globHit.size(); ++hit){
    hitPhi.push_back(static_cast<double>(globHit[hit].phi()));
    // localPositionError[hit] or assume 0.003, 0.006
    hitPhiError.push_back( 0.003 / globHit[hit].perp());
    // no errors on z, fill with zeros
    hitZprimeError.push_back(0.0);
    // barrel-specific values
    if(beam.isAlignmentTube() && abs(globHit[hit].z()) < 112.3){
      gBarrelModuleRadius.push_back(globHit[hit].perp());
      gBarrelModuleOffset.push_back(gBarrelModuleRadius.back() - gBeamR);
      // TIB/TOB flag
      if(gBarrelModuleOffset.back() < 0.0){
	gIsInnerBarrel = 1;
      }
      else{
	gIsInnerBarrel = 0;
      }
      gHitZprime.push_back(globHit[hit].z() - gBeamZ0 - abs(gBarrelModuleOffset.back())); 
    }
    // non-barrel z'
    else{
      gHitZprime.push_back(globHit[hit].z() - gBeamZ0);
    }
  }

  // number of fit parameters, 3 for TECs (always!); 3, 5, or 6 for ATs
  unsigned int tecParams(3), atParams(0);
  if(nAtParameters_ == 3) atParams = 3;
  else if(nAtParameters_ == 5) atParams = 5;
  else atParams = 6;                            // <-- default value, recommended
  unsigned int nFitParams(0);
  if(!fitBeamSplitters_ || 
     (hitsAtTecPlus == 0 && beam.isAlignmentTube() ) ){
    tecParams = tecParams - 1;
    atParams = atParams - 1;
  }
  if(beam.isTecInternal()){
    nFitParams = tecParams;
  }
  else{
    nFitParams = atParams;
  }
  
  // fit parameter definitions
  double offset(0.), offsetError(0.), slope(0.), slopeError(0.),
    bsAngleParam(0.), phiAtMinusParam(0.), phiAtPlusParam(0.),
    atThetaSplitParam(0.);
  AlgebraicSymMatrix covMatrix;
  if(!fitBeamSplitters_ || (beam.isAlignmentTube() && hitsAtTecPlus == 0)){
    covMatrix = AlgebraicSymMatrix(nFitParams, 1);
  }
  else{
    covMatrix = AlgebraicSymMatrix(nFitParams - 1, 1);
  }

  this->fitter(beam, covMatrix, 
	       hitsAtTecPlus, nFitParams, 
	       hitPhi, hitPhiError, hitZprimeError, 
	       zMin, zMax, bsAngleParam,
	       offset, offsetError, slope, slopeError,
	       phiAtMinusParam, phiAtPlusParam,
	       atThetaSplitParam); 
  
  vector<GlobalPoint> globPtrack;
  GlobalPoint globPref;
  double chi2(0.);

  for(unsigned int hit = 0; hit < gHitZprime.size(); ++hit){
    
    // additional phi value (trackPhiRef) for trajectory calculation
    double trackPhi(0.), trackPhiRef(0.); 
    
    this->trackPhi(beam, hit, trackPhi, trackPhiRef,
		   offset, slope, bsAngleParam, 
		   phiAtMinusParam, phiAtPlusParam,
		   atThetaSplitParam, globHit);

    cout << "track phi = " << trackPhi 
	 << ", hit phi = " << hitPhi[hit] 
	 << ", zPrime = " << gHitZprime[hit] 
	 << " r = " << globHit[hit].perp() << endl;

    this->globalTrackPoint(beam, hit, hitsAtTecPlus, 
			   trackPhi, trackPhiRef, 
			   globHit, globPtrack, globPref, 
			   hitPhiError);

    // calculate residuals = pred - hit (in global phi)
    const double phiResidual = globPtrack[hit].phi() - globHit[hit].phi();
    // pull calculation (FIX!!!)
    const double phiResidualPull = phiResidual / hitPhiError[hit];
    //       sqrt(hitPhiError[hit]*hitPhiError[hit] + 
    // 	   (offsetError*offsetError + globPtrack[hit].z()*globPtrack[hit].z() * slopeError*slopeError));

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
  }

  cout << "chi^2 = " << chi2 << ", chi^2/ndof = " << chi2/(gHitZprime.size() - nFitParams) << endl;
  this->fitBeam(beam, covMatrix, hitsAtTecPlus, nFitParams,
		offset, slope, globPtrack, bsAngleParam, chi2);
  
  cout << "bsAngleParam = " << bsAngleParam << endl;

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
void TkLasBeamFitter::getLasHits(TkFittedLasBeam &beam, const SiStripLaserRecHit2D &hit, 
				 vector<const GeomDetUnit*> &gd, vector<GlobalPoint> &globHit,
				 unsigned int &hitsAtTecPlus)
{ 
  // get global position of LAS hits
  gd.push_back(geometry->idToDetUnit(hit.getDetId()));
  GlobalPoint globPtemp(gd.back()->toGlobal(hit.localPosition()));

  // testing: globPtemp should be right
  globHit.push_back(globPtemp);

  if(beam.isAlignmentTube()){
    if(abs(globPtemp.z()) > 112.3){
      if(globPtemp.z() < 112.3) gHitsAtTecMinus++ ;
      else hitsAtTecPlus++ ;
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
  // BarrelMinus
  else if(-gBeamSplitterZprime - 2.0*gBeamZ0 < z && z < -gBeamZ0){
    // z value includes module offset from main beam axis
    // TOB
    if(!gIsInnerBarrel){
      return par[0] + par[1] * z + gTOBparam * (par[2] + par[4]);
    }
    // TIB
    else{
      return par[0] + par[1] * z - gTIBparam * (par[2] - par[4]);
    }
  }
  // BarrelPlus
  else if(-gBeamZ0 < z && z < gBeamSplitterZprime){
    // z value includes module offset from main beam axis
    // TOB
    if(!gIsInnerBarrel){
      return par[0] + par[1] * z + gTOBparam * (par[3] - par[4]);
    }
    // TIB
    else{
      return par[0] + par[1] * z - gTIBparam * (par[3] + par[4]);
    }
  }
  // TECplus
  else{
    if(gFitBeamSplitters){
      // par[2] = 2*tan(BeamSplitterAngle/2.0)
      return par[0] + par[1] * z - par[5] * (z - gBeamSplitterZprime)/gBeamR; // BS par: 5, 4, or 2
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
			     double &offset, double &offsetError, double &slope, double &slopeError,
			     double &phiAtMinusParam, double &phiAtPlusParam, double &atThetaSplitParam)
{
  TGraphErrors *lasData = new TGraphErrors(gHitZprime.size(), 
					   &(gHitZprime[0]), &(hitPhi[0]), 
					   &(hitZprimeError[0]), &(hitPhiError[0]));
  
  // do fit (R = entire range)
  if(beam.isTecInternal(1)){
    TF1 tecPlus("tecPlus", tecPlusFunction, zMin, zMax, nFitParams );
    tecPlus.SetParameter( 1, 0 ); // slope
    tecPlus.SetParameter( nFitParams - 1, 0 ); // BS 
    lasData->Fit("tecPlus", "R"); // "R", "RV" or "RQ"
  }
  else if(beam.isTecInternal(-1)){
    TF1 tecMinus("tecMinus", tecMinusFunction, zMin, zMax, nFitParams );
    tecMinus.SetParameter( 1, 0 ); // slope
    tecMinus.SetParameter( nFitParams - 1, 0 ); // BS 
    lasData->Fit("tecMinus", "R");
  }
  else{
    TF1 at("at", atFunction, zMin, zMax, nFitParams );
    at.SetParameter( 1, 0 ); // slope
    at.SetParameter( nFitParams - 1, 0 ); // BS 
    lasData->Fit("at","R");
  }
  
  // get values and errors for offset and slope
  gMinuit->GetParameter(0, offset, offsetError);
  gMinuit->GetParameter(1, slope, slopeError);

  // additional AT parameters
  // define param errors that are not used later
  double bsAngleParamError(0.), phiAtMinusParamError(0.),
    phiAtPlusParamError(0.), atThetaSplitParamError(0.);

  if(beam.isAlignmentTube()){
    gMinuit->GetParameter(2, phiAtMinusParam, phiAtMinusParamError);
    gMinuit->GetParameter(3, phiAtPlusParam, phiAtPlusParamError);
    gMinuit->GetParameter(4, atThetaSplitParam, atThetaSplitParamError);
  }
  // get Beam Splitter parameters
  if(fitBeamSplitters_){
    if(beam.isAlignmentTube() && hitsAtTecPlus == 0){
      bsAngleParam = gBSparam;
    }
    else{
      gMinuit->GetParameter( nFitParams - 1 , bsAngleParam, bsAngleParamError);
    }
  }
  else{
    bsAngleParam = gBSparam;
  }

  // fill covariance matrix
  vector<double> vec( covMatrix.num_col() * covMatrix.num_col() );
  gMinuit->mnemat( &vec[0], covMatrix.num_col() );
  for(int col = 0; col < covMatrix.num_col(); col++){
    for(int row = 0; row < covMatrix.num_col(); row++){
      covMatrix[col][row] = vec[row + covMatrix.num_col()*col];
    }
  }
  // compute correlation between parameters
//   double corr01 = covMatrix[1][0]/(offsetError*slopeError);

  delete lasData;
}


// -------------- calculate track phi value ----------------------------------
void TkLasBeamFitter::trackPhi(TkFittedLasBeam &beam, unsigned int &hit, 
			       double &trackPhi, double &trackPhiRef,
			       double &offset, double &slope, double &bsAngleParam,
			       double &phiAtMinusParam, double &phiAtPlusParam,
			       double &atThetaSplitParam, vector<GlobalPoint> &globHit)
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
    // BarrelMinus
    else if(-gBeamSplitterZprime - 2.0*gBeamZ0 < gHitZprime[hit] && gHitZprime[hit] < -gBeamZ0){
      if(!gIsInnerBarrel){
	trackPhi = offset + slope * gHitZprime[hit] + gTOBparam * (phiAtMinusParam + atThetaSplitParam); 
      }
      else{
	trackPhi = offset + slope * gHitZprime[hit] - gTIBparam * (phiAtMinusParam - atThetaSplitParam);
      }
      trackPhiRef = offset + slope * (gHitZprime[hit] + abs(gBarrelModuleOffset[hit-gHitsAtTecMinus]));
    }
    // BarrelPlus
    else if(-gBeamZ0 < gHitZprime[hit] && gHitZprime[hit] < gBeamSplitterZprime){
      if(!gIsInnerBarrel){
	trackPhi = offset + slope * gHitZprime[hit] + gTOBparam * (phiAtPlusParam - atThetaSplitParam);
      }
      else{
	trackPhi = offset + slope * gHitZprime[hit] - gTIBparam * (phiAtPlusParam + atThetaSplitParam);
      }
      trackPhiRef = offset + slope * (gHitZprime[hit] + abs(gBarrelModuleOffset[hit-gHitsAtTecMinus]));
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
  unsigned int paramType(0);
  if(!fitBeamSplitters_) paramType = 1;
  if(beam.isAlignmentTube() && hitsAtTecPlus == 0) paramType = 0;
//   const unsigned int nPedeParams = nFitParams + paramType;

  // test without BS params
  const unsigned int nPedeParams(nFitParams);
//   cout << "number of Pede parameters: " << nPedeParams << endl;

  std::vector<TkFittedLasBeam::Scalar> params(nPedeParams);
  params[0] = offset;
  params[1] = slope;
  // no BS parameter for AT beams without TECplus hits
//   if(beam.isTecInternal() || hitsAtTecPlus > 0) params[2] = bsAngleParam;

  AlgebraicMatrix derivatives(gHitZprime.size(), nPedeParams);
  // fill derivatives matrix with local track derivatives
  for(unsigned int hit = 0; hit < gHitZprime.size(); ++hit){
    
    // d(delta phi)/d(offset) is identical for every hit
    derivatives[hit][0] = 1.0;

    // d(delta phi)/d(slope) and d(delta phi)/d(bsAngleParam) depend on parametrizations
    // TECplus
    if(beam.isTecInternal(1)){
      derivatives[hit][1] = globPtrack[hit].z();
//       if(gHitZprime[hit] < gBeamSplitterZprime){
// 	derivatives[hit][2] = 0.0;
//       }
//       else{
// 	derivatives[hit][2] = - (globPtrack[hit].z() - gBeamSplitterZprime) / gBeamR;
//       }
    }
    // TECminus
    else if(beam.isTecInternal(-1)){
      derivatives[hit][1] = globPtrack[hit].z();
//       if(gHitZprime[hit] > gBeamSplitterZprime){
// 	derivatives[hit][2] = 0.0;
//       }
//       else{
// 	derivatives[hit][2] = (globPtrack[hit].z() - gBeamSplitterZprime) / gBeamR;
//       }
    }
    // ATs
    else{
      // TECminus
      if(gHitZprime[hit] < -gBeamSplitterZprime - 2.0*gBeamZ0){
	derivatives[hit][1] = globPtrack[hit].z();
// 	if(hitsAtTecPlus > 0){
// 	  derivatives[hit][2] = 0.0;
// 	}
      }
      // TECplus
      else if(gHitZprime[hit] > gBeamSplitterZprime){
	derivatives[hit][1] = globPtrack[hit].z();
// 	if(hitsAtTecPlus > 0){
// 	  derivatives[hit][2] = - (globPtrack[hit].z() - gBeamSplitterZprime) / gBeamR;
// 	}
      }
      // Barrel
      else{
	derivatives[hit][1] = globPtrack[hit].z() - gBarrelModuleOffset[hit-gHitsAtTecMinus];
// 	if(hitsAtTecPlus > 0){
// 	  derivatives[hit][2] = 0.0;
// 	}
      }
    }
  }

  unsigned int firstFixedParam(covMatrix.num_col()); // FIXME! --> no, is fine!!!
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
