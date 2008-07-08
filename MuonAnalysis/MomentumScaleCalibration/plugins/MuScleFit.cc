//  \class MuScleFit
//  Analyzer of the StandAlone muon tracks
//
//  $Date: 2008/07/08 10:33:51 $
//  $Revision: 1.3 $
//  \author R. Bellan, C.Mariotti, S.Bolognesi - INFN Torino / T.Dorigo, M.De Mattia - INFN Padova
//
//  Recent additions: 
//  - several parameters allow a more flexible use, tests, and control handles
//    for the likelihood. In particular, a set of integers controls the order
//    with which parameters are released; another controls which parameters are
//    fixed. A function allows to smear momenta and angles of muons from the
//    resonance before any correction, using a set of random numbers generated
//    once and for all in the constructor (this way the smearing remains the same
//    for any given muon no matter how many times one loops and what corrections
//    one applies).
//    For a correct use of these flags, please see the function minimizeLikelihood() in
//    MuScleFitUtils.cc
//  - the fit now allows to extract resolution functions simultaneously with the
//    momentum scale. So far a simple parametrization
//    of muon momentum resolution and angle resolution has been implemented, but
//    extensions are straightforward.
//  - It is however advisable to fit separately resolution and scale. The suggested
//    course of action is: 
//    1) fit the scale with a simple parametrization
//    2) check results, fit with more complicated forms
//    3) verify which is a sufficiently accurate description of the data
//    4) fix scale parameters and fit for the resolution
//    5) go back to fitting the scale with resolution parameters fixed to fitted values
//  - Also note that resolution fits may fail to converge due to instability of the
//    probability distribution to the limit of large widths. Improvements here are
//    advisable.
//  - The treatment of signal windows in the Y region
//    has to be refined because of overlaps. More work is needed here, assigning a different
//    weight to different hypothesis of the resonance producing a given mass, if there are
//    multiple candidates. 
//  - Also, larger windows are to be allowed for fits to SA muons.
//  - File Probs_1000.root contains the probability distribution of lorentzians convoluted
//    with gaussian smearing functions, for the six resonances. A 1000x1000 grid
//    in mass,sigma has been computed (using root macro Probs.C). 
//    A wider interval of masses for each resonance should be computed, to be used for standalone muons
//
//
//  Notes on additions, TD 31/3/08
//
//  - background model: at least a couple of different models, with two parameters,
//    should be included in the fitting procedure such that the function massprob(),
//    which produces the probability in the likelihood computation, incorporates the
//    probability that the event is from background. That is, if the fitting function
//    knows the shape of the mass spectrum ( say, a falling exponential plus a gaussian
//    signal) it becomes possible to fit the scale together with the background shape
//    and normalization parameters. Of course, one should do one thing at a time: first
//    a scale fit, then a shape fit with scale parameters fixed, and then a combined
//    fit. Convergence issues should be handled case by case.
//  - The correct implementation of the above idea requires a reorganization of pass
//    parameters (in the cfg) and fit parameters. The user has to be able to smear,
//    bias, fix parameters, choose scale fitting functions, resolution fitting functions,
//    and background functions. It should be possible to separate the fit functions from
//    the biasing ones, which would allow a more thorough testing.
//  - all the above can be obtained by making the .cfg instructions heavier. Since this
//    is a routine operated by experts only, it is a sensible choice.
//  - One should thus envision the following:
//      1) a set of parameters controlling the biasing function: parBias()
//      2) a set of parameters controlling the smearing function: parSmear()
//      3) a set of parameters to define resolution modeling and initial values: parResol()
//      3b) parResol() gets fix and order bits by parResolFix() and parResolOrder()
//      4) a set of parameters to define scale modeling and initial values: parScale()
//      4b) parScale() gets fix and order bits by parScaleFix() and parScaleOrder()
//      5) a set of parameters controlling the background shape and normalization: parNorm()
//      5b) parNorm() gets fix and order bits by parNormFix() and parNormOrder()
//    The likelihood parameters then become a vector which is dynamically composed of
//    sets 3), 4), and 5): parval() = parResol()+parScale()+parNorm()
//  - In order to study better the likelihood behavior it would be advisable to introduce
//    some histogram filling on the last iteration of the likelihood. It is not clear
//    how best to achieve that: probably the simplest way is to make a histogram filling
//    function run just after the likelihood computation, such that the best value of the
//    fit parameters is used.
//  - The muon pair which we call our resonance must be chosen in a way which does not
//    bias our likelihood: we cannot just choose the pair closest to a resonance.
// ----------------------------------------------------------------------------------

#include "MuScleFit.h"
#include "Histograms.h"
#include "MuScleFitUtils.h"
#include "MuScleFitPlotter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "HepPDT/defs.h"
#include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"

#include "HepMC/GenParticle.h"
#include "HepMC/GenEvent.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
// The following is required in CMSSW v2.0.x (was contained in Muon.h in 1.6.7)
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include <CLHEP/Vector/LorentzVector.h>

#include "TFile.h"
#include "TTree.h"
#include "TMinuit.h"
#include <vector>

using namespace std;
using namespace edm;
using namespace reco; // For AODSIM MC objects

// Constructor
// -----------
MuScleFit::MuScleFit (const ParameterSet& pset) {

  debug = pset.getUntrackedParameter<int>("debug",0);
  MuScleFitUtils::debug = debug;
  if (debug>0) cout << "[MuScleFit]: Constructor" << endl;

  // Service parameters
  // ------------------
  ParameterSet serviceParameters = pset.getParameter<ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);  
  theMuonLabel    = pset.getParameter<InputTag>("MuonLabel");
  theMuonType     = pset.getParameter<int>("muonType");
  theRootFileName = pset.getUntrackedParameter<string>("RootFileName", "GLBMuonHistos.root");

  if (theMuonType<1 || theMuonType>3) {
    cout << "[MuScleFit]: Unknown muon type! Aborting." << endl;
    abort();
  }  

  loopCounter = 0;

  // Boundaries for h-function computation (to be improved!)
  // -------------------------------------------------------
  minResMass_hwindow[0] = 76.;   
  maxResMass_hwindow[0] = 106.;  
  minResMass_hwindow[1] = 10.15; 
  maxResMass_hwindow[1] = 10.55;
  minResMass_hwindow[2] = 9.8; 
  maxResMass_hwindow[2] = 10.2;
  minResMass_hwindow[3] = 9.25; 
  maxResMass_hwindow[3] = 9.65;
  minResMass_hwindow[4] = 3.58; 
  maxResMass_hwindow[4] = 3.78;
  minResMass_hwindow[5] = 3.0; 
  maxResMass_hwindow[5] = 3.2;

  // Max number of loops (if > 2 then try to minimize likelihood more than once)
  // ---------------------------------------------------------------------------
  maxLoopNumber = pset.getUntrackedParameter<int>("maxLoopNumber", 2);

  // Bias and smear types
  // --------------------
  MuScleFitUtils::BiasType  = pset.getParameter<int>("BiasType");
  MuScleFitUtils::SmearType = pset.getParameter<int>("SmearType");

  // Fit types
  // ---------
  MuScleFitUtils::ResolFitType = pset.getParameter<int>("ResolFitType");
  MuScleFitUtils::ScaleFitType = pset.getParameter<int>("ScaleFitType");
  MuScleFitUtils::BgrFitType   = pset.getParameter<int>("BgrFitType");

  // Initial parameters values
  // -------------------------
  MuScleFitUtils::parBias  = pset.getParameter<vector<double> >("parBias");
  MuScleFitUtils::parSmear = pset.getParameter<vector<double> >("parSmear");
  MuScleFitUtils::parResol = pset.getParameter<vector<double> >("parResol");
  MuScleFitUtils::parScale = pset.getParameter<vector<double> >("parScale");
  MuScleFitUtils::parBgr   = pset.getParameter<vector<double> >("parBgr");
  MuScleFitUtils::parResolFix   = pset.getParameter<vector<int> >("parResolFix");
  MuScleFitUtils::parScaleFix   = pset.getParameter<vector<int> >("parScaleFix");
  MuScleFitUtils::parBgrFix     = pset.getParameter<vector<int> >("parBgrFix");
  MuScleFitUtils::parResolOrder = pset.getParameter<vector<int> >("parResolOrder");
  MuScleFitUtils::parScaleOrder = pset.getParameter<vector<int> >("parScaleOrder");
  MuScleFitUtils::parBgrOrder   = pset.getParameter<vector<int> >("parBgrOrder");

  MuScleFitUtils::resfind     = pset.getParameter<vector<int> >("resfind");
  MuScleFitUtils::FitStrategy = pset.getParameter<int>("FitStrategy");

  // Option to skip unnecessary stuff
  // --------------------------------
  MuScleFitUtils::speedup = pset.getParameter<bool>("speedup");

  // Bias parameters: dimension check
  // --------------------------------
  if ((MuScleFitUtils::BiasType==1  && MuScleFitUtils::parBias.size()!=2) || // linear in pt
      (MuScleFitUtils::BiasType==2  && MuScleFitUtils::parBias.size()!=2) || // linear in |eta|
      (MuScleFitUtils::BiasType==3  && MuScleFitUtils::parBias.size()!=2) || // sinusoidal in phi
      (MuScleFitUtils::BiasType==4  && MuScleFitUtils::parBias.size()!=3) || // linear in pt and |eta|
      (MuScleFitUtils::BiasType==5  && MuScleFitUtils::parBias.size()!=3) || // linear in pt and sinusoidal in phi
      (MuScleFitUtils::BiasType==6  && MuScleFitUtils::parBias.size()!=3) || // linear in |eta| and sinusoidal in phi
      (MuScleFitUtils::BiasType==7  && MuScleFitUtils::parBias.size()!=4) || // linear in pt and |eta| and 
                                                                                         // sinusoidal in phi
      (MuScleFitUtils::BiasType==8  && MuScleFitUtils::parBias.size()!=4) || // linear in pt and parabolic in |eta|
      (MuScleFitUtils::BiasType==9  && MuScleFitUtils::parBias.size()!=2) || // exponential in pt
      (MuScleFitUtils::BiasType==10 && MuScleFitUtils::parBias.size()!=3) || // parabolic in pt
      (MuScleFitUtils::BiasType==11 && MuScleFitUtils::parBias.size()!=4) || // linear in pt and sin in phi with chg
      (MuScleFitUtils::BiasType==12 && MuScleFitUtils::parBias.size()!=6) || // linear in pt and para in eta 
                                                                                         // plus sin in phi with chg
      (MuScleFitUtils::BiasType==13 && MuScleFitUtils::parBias.size()!=8) || // linear in pt and para in eta 
                                                                                         // plus sin in phi with chg
      MuScleFitUtils::BiasType<0 || MuScleFitUtils::BiasType>13) {
    cout << "[MuScleFit-Constructor]: Wrong bias type or number of parameters: aborting!" << endl;
    abort();
  }
  // Smear parameters: dimension check
  // ---------------------------------
  if ((MuScleFitUtils::SmearType==1  && MuScleFitUtils::parSmear.size()!=3) || 
      (MuScleFitUtils::SmearType==2  && MuScleFitUtils::parSmear.size()!=4) ||
      (MuScleFitUtils::SmearType==3  && MuScleFitUtils::parSmear.size()!=5) ||
      (MuScleFitUtils::SmearType==4  && MuScleFitUtils::parSmear.size()!=6) ||
      (MuScleFitUtils::SmearType==5  && MuScleFitUtils::parSmear.size()!=7) ||
      MuScleFitUtils::SmearType<0 || MuScleFitUtils::SmearType>5) {
    cout << "[MuScleFit-Constructor]: Wrong smear type or number of parameters: aborting!" << endl;
    abort();
  }

  // Resol fit parameters: dimension check
  // -------------------------------------
  if ((MuScleFitUtils::ResolFitType==1 && MuScleFitUtils::parResol.size()!=3) ||
      (MuScleFitUtils::ResolFitType==2 && MuScleFitUtils::parResol.size()!=4) ||
      (MuScleFitUtils::ResolFitType==3 && MuScleFitUtils::parResol.size()!=5) ||
      (MuScleFitUtils::ResolFitType==4 && MuScleFitUtils::parResol.size()!=6) ||
      (MuScleFitUtils::ResolFitType==5 && MuScleFitUtils::parResol.size()!=7) ||
      MuScleFitUtils::ResolFitType<1 || MuScleFitUtils::ResolFitType>5) {
    cout << "[MuScleFit-Constructor]: Wrong Resol fit type or number of parameters: aborting!" << endl;
    abort();
  }
  // Scale fit parameters: dimension check
  // -------------------------------------
  if ((MuScleFitUtils::ScaleFitType==1  && MuScleFitUtils::parScale.size()!=2) || // linear in pt
      (MuScleFitUtils::ScaleFitType==2  && MuScleFitUtils::parScale.size()!=2) || // linear in |eta|
      (MuScleFitUtils::ScaleFitType==3  && MuScleFitUtils::parScale.size()!=2) || // sinusoidal in phi
      (MuScleFitUtils::ScaleFitType==4  && MuScleFitUtils::parScale.size()!=3) || // linear in pt and |eta|
      (MuScleFitUtils::ScaleFitType==5  && MuScleFitUtils::parScale.size()!=3) || // linear in pt and sinusoidal in phi
      (MuScleFitUtils::ScaleFitType==6  && MuScleFitUtils::parScale.size()!=3) || // linear in |eta| and 
                                                                                              // sinusoidal in phi
      (MuScleFitUtils::ScaleFitType==7  && MuScleFitUtils::parScale.size()!=4) || // linear in pt and |eta| and 
                                                                                              // sinusoidal in phi
      (MuScleFitUtils::ScaleFitType==8  && MuScleFitUtils::parScale.size()!=4) || // linear in pt and parabolic in |eta|
      (MuScleFitUtils::ScaleFitType==9  && MuScleFitUtils::parScale.size()!=2) || // exponential in pt
      (MuScleFitUtils::ScaleFitType==10 && MuScleFitUtils::parScale.size()!=3) || // parabolic in pt
      (MuScleFitUtils::ScaleFitType==11 && MuScleFitUtils::parScale.size()!=4) || // linear in pt and sin in phi w/ chg
      (MuScleFitUtils::ScaleFitType==12 && MuScleFitUtils::parScale.size()!=6) || // linear in pt and para in eta 
                                                                                              // plus sin in phi with chg
      (MuScleFitUtils::ScaleFitType==13 && MuScleFitUtils::parScale.size()!=8) || // linear in pt and para in eta 
                                                                                              // plus sin in phi with chg
      MuScleFitUtils::ScaleFitType<1 || MuScleFitUtils::ScaleFitType>13) {
    cout << "[MuScleFit-Constructor]: Wrong fit type or number of parameters: aborting!" << endl;
    abort();
  }
  // Bgr fit parameters: dimension check
  // -----------------------------------
  if ((MuScleFitUtils::BgrFitType==1 && MuScleFitUtils::parBgr.size()!=1) ||
      (MuScleFitUtils::BgrFitType==2 && MuScleFitUtils::parBgr.size()!=2) ||
      (MuScleFitUtils::BgrFitType==3 && MuScleFitUtils::parBgr.size()!=3) ||
      MuScleFitUtils::BgrFitType<1 || MuScleFitUtils::BgrFitType>3) {
    cout << "[MuScleFit-Constructor]: Wrong Bgr fit type or number of parameters: aborting!" << endl;
    abort();
  }

  // Protect against bad size of parameters
  // --------------------------------------
  if (MuScleFitUtils::parResol.size()!=MuScleFitUtils::parResolFix.size() || 
      MuScleFitUtils::parResol.size()!=MuScleFitUtils::parResolOrder.size()) {
    cout << "[MuScleFit-Constructor]: Mismatch in number of parameters for Resol: aborting!" << endl;
    abort();
  }
  if (MuScleFitUtils::parScale.size()!=MuScleFitUtils::parScaleFix.size() || 
      MuScleFitUtils::parScale.size()!=MuScleFitUtils::parScaleOrder.size()) {
    cout << "[MuScleFit-Constructor]: Mismatch in number of parameters for Scale: aborting!" << endl;
    abort();
  }
  if (MuScleFitUtils::parBgr.size()!=MuScleFitUtils::parBgrFix.size() || 
      MuScleFitUtils::parBgr.size()!=MuScleFitUtils::parBgrOrder.size()) {
    cout << "[MuScleFit-Constructor]: Mismatch in number of parameters for Bgr: aborting!" << endl;
    abort();
  }
  
  // Protect against an incorrect number of resonances
  // -------------------------------------------------
  if (MuScleFitUtils::resfind.size()!=6) {
    cout << "[MuScleFit-Constructor]: resfind must have 6 elements (1 Z, 3 Y, 2 Psi): aborting!" << endl;
    abort();
  }

  // Generate array of gaussian-distributed numbers for smearing
  // -----------------------------------------------------------
  if (MuScleFitUtils::SmearType>0) {
    cout << "[MuScleFit-Constructor]: Generating random values for smearing" << endl;
    TF1 * G = new TF1 ("G", "[0]*exp(-0.5*pow(x,2))", -5., 5.);
    double norm = 1/sqrt(2*TMath::Pi());
    G->SetParameter (0,norm);    
    for (int i=0; i<10000; i++) {
      for (int j=0; j<7; j++) {
	MuScleFitUtils::x[j][i] = G->GetRandom();
      }
    }
  }
  MuScleFitUtils::goodmuon = 0;

  // Read probability distributions from root file
  // These are 2-D PDFs containing a grid of 1000x1000 values of the
  // integral of Lorentz * Gaussian as a function
  // of mass and resolution of a given measurement,
  // for each of the six considered diLmuon resonances.
  // -------------------------------------------------
  cout << "[MuScleFit-Constructor]: Reading TH2D probabilities from Probs_1000.root file" << endl;
  TH2D * GL[6];
  TFile * ProbsFile;
  if ( theMuonType!=2 ) {
    ProbsFile = new TFile ("Probs_1000.root"); // NNBB need to reset this if MuScleFitUtils::nbins changes
  } else {
    ProbsFile = new TFile ("Probs_SM_1000.root"); // NNBB need to reset this if MuScleFitUtils::nbins changes
  }    
  ProbsFile->cd();
  GL[0] = dynamic_cast<TH2D*> (ProbsFile->Get("GL0"));
  GL[1] = dynamic_cast<TH2D*> (ProbsFile->Get("GL1"));
  GL[2] = dynamic_cast<TH2D*> (ProbsFile->Get("GL2"));
  GL[3] = dynamic_cast<TH2D*> (ProbsFile->Get("GL3"));
  GL[4] = dynamic_cast<TH2D*> (ProbsFile->Get("GL4"));
  GL[5] = dynamic_cast<TH2D*> (ProbsFile->Get("GL5"));

  // Extract normalization for each mass slice
  // -----------------------------------------
  for (int ires=0; ires<6; ires++) {
    for (int iy=0; iy<=MuScleFitUtils::nbins; iy++) {
      MuScleFitUtils::GLNorm[ires][iy] = 0.;
      for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
	MuScleFitUtils::GLValue[ires][ix][iy] = GL[ires]->GetBinContent (ix+1, iy+1);
	MuScleFitUtils::GLNorm[ires][iy] += MuScleFitUtils::GLValue[ires][ix][iy];
      }
      if (debug>2) cout << "GLValue[" << ires << "][500][" << iy << "] = " 
		       << MuScleFitUtils::GLValue[ires][500][iy] 
		       << " GLNorm[" << ires << "][" << iy << "] = " 
		       << MuScleFitUtils::GLNorm[ires][iy] << endl;
    }
  }
  // Initialize ResMaxSigma And ResHalfWidth - 0 = global, 1 = SM, 2 = tracker
  // -------------------------------------------------------------------------
  MuScleFitUtils::ResMaxSigma[0][0] = 50.;
  MuScleFitUtils::ResMaxSigma[1][0] = 5.;
  MuScleFitUtils::ResMaxSigma[2][0] = 5.;
  MuScleFitUtils::ResMaxSigma[3][0] = 5.;
  MuScleFitUtils::ResMaxSigma[4][0] = 2.;
  MuScleFitUtils::ResMaxSigma[5][0] = 2.;
  MuScleFitUtils::ResMaxSigma[0][1] = 100.;
  MuScleFitUtils::ResMaxSigma[1][1] = 10.;
  MuScleFitUtils::ResMaxSigma[2][1] = 10.;
  MuScleFitUtils::ResMaxSigma[3][1] = 10.;
  MuScleFitUtils::ResMaxSigma[4][1] = 5.;
  MuScleFitUtils::ResMaxSigma[5][1] = 5.;
  MuScleFitUtils::ResMaxSigma[0][2] = 50.;
  MuScleFitUtils::ResMaxSigma[1][2] = 5.;
  MuScleFitUtils::ResMaxSigma[2][2] = 5.;
  MuScleFitUtils::ResMaxSigma[3][2] = 5.;
  MuScleFitUtils::ResMaxSigma[4][2] = 2.;
  MuScleFitUtils::ResMaxSigma[5][2] = 2.;
  MuScleFitUtils::ResHalfWidth[0][0] =20.;
  MuScleFitUtils::ResHalfWidth[1][0] = 0.5;
  MuScleFitUtils::ResHalfWidth[2][0] = 0.5;
  MuScleFitUtils::ResHalfWidth[3][0] = 0.5;
  MuScleFitUtils::ResHalfWidth[4][0] = 0.2;
  MuScleFitUtils::ResHalfWidth[5][0] = 0.2;
  MuScleFitUtils::ResHalfWidth[0][1] =50.;
  MuScleFitUtils::ResHalfWidth[1][1] = 2.5;
  MuScleFitUtils::ResHalfWidth[2][1] = 2.5;
  MuScleFitUtils::ResHalfWidth[3][1] = 2.5;
  MuScleFitUtils::ResHalfWidth[4][1] = 1.5;
  MuScleFitUtils::ResHalfWidth[5][1] = 1.5;
  MuScleFitUtils::ResHalfWidth[0][2] =20.;
  MuScleFitUtils::ResHalfWidth[1][2] = 0.5;
  MuScleFitUtils::ResHalfWidth[2][2] = 0.5;
  MuScleFitUtils::ResHalfWidth[3][2] = 0.5;
  MuScleFitUtils::ResHalfWidth[4][2] = 0.2;
  MuScleFitUtils::ResHalfWidth[5][2] = 0.2;

  MuScleFitUtils::MuonType = theMuonType-1;

}

// Destructor
// ----------
MuScleFit::~MuScleFit () {
  if (debug>0) cout << "[MuScleFit]: Destructor" << endl;
}

// Begin job
// ---------
void MuScleFit::beginOfJob (const EventSetup& eventSetup) {

  if (debug>0) cout << "[MuScleFit]: beginOfJob" << endl;
  
  // Create the root file
  // --------------------
  for (unsigned int i=0; i<(maxLoopNumber); i++) {
    char buffer [2]; // FIXME: the max number of loop has to be < 10, (otherwise not enough char)
    sprintf (buffer, "%d_", i);
    string rootFileName = buffer + theRootFileName ;
    theFiles.push_back (new TFile(rootFileName.c_str(), "RECREATE"));
  }
  if (debug>0) cout << "[MuScleFit]: Root file created" << endl;

  plotter = new MuScleFitPlotter();
  plotter->debug = debug; 
}

// End of job method
// -----------------
void MuScleFit::endOfJob () {
  if (debug>0) cout << "[MuScleFit]: endOfJob" << endl;
  delete plotter;
}

// New loop
// --------
void MuScleFit::startingNewLoop (unsigned int iLoop) {

  if (debug>0) cout << "[MuScleFit]: Starting loop # " << iLoop << endl;

  // Number of muons used 
  // --------------------
  MuScleFitUtils::goodmuon = 0;

  // Counters for problem cout-ing
  // -----------------------------
  MuScleFitUtils::counter_resprob = 0;

  // Create the root file
  // --------------------
  theFiles[iLoop]->cd();
  fillHistoMap();

  loopCounter = iLoop;
  MuScleFitUtils::loopCounter = loopCounter;

  iev = 0;

  MuScleFitUtils::cleanEstimator();

}

// End of loop routine
// -------------------
edm::EDLooper::Status MuScleFit::endOfLoop (const edm::EventSetup& eventSetup, unsigned int iLoop) {

  cout << "Ending loop # " << iLoop << endl;
  cout << "Number of events with Z after ewk cuts" << numberOfEwkZ << endl;
  if (debug>0) {
    cout << "Number of Sim tracks:   " << numberOfSimTracks << endl;
    cout << "Number of Sim muons:    " << numberOfSimMuons << endl;
    cout << "Number of Sim vertices: " << numberOfSimVertices << endl;
  }

  // Write the histos to file
  // ------------------------
  theFiles[iLoop]->cd();
  writeHistoMap();

  // Mass probability histograms
  // ---------------------------
  Mass_P->Write();
  Mass_fine_P->Write();

  theFiles[iLoop]->Close();

  // Likelihood minimization to compute corrections
  // ----------------------------------------------
  MuScleFitUtils::minimizeLikelihood();

  // Clear the histos
  // ----------------
  clearHistoMap();

  //Compute the estimator h
  //-----------------------
  if (!MuScleFitUtils::speedup) MuScleFitUtils::returnEstimator();
 
  if (iLoop>=maxLoopNumber-1) {
    return kStop;
  } else {
    return kContinue;
  }
}

// Stuff to do during loop
// -----------------------
edm::EDLooper::Status MuScleFit::duringLoop (const Event & event, const EventSetup& eventSetup) {

  // Update the services
  // -------------------
  theService->update (eventSetup);
  if (debug>0) {
    cout << "[MuScleFit-duringLoop]: loopCounter = " << loopCounter
	 << " Run: " << event.id().run() << " Event: " << event.id().event() << endl;
  }

  // Find and store in histograms the generated and simulated resonance and muons
  // ----------------------------------------------------------------------------
  if (!MuScleFitUtils::speedup) { // NB we skip the simulation part if we are in a hurry
    if(loopCounter==0){                           //Only at the first loop

      bool ifGen=false;
      Handle<HepMCProduct> evtMC;
      try {
	event.getByLabel ("source", evtMC);
	plotter->fillGen2(evtMC);
	ifGen=true;
      } catch (...) { 
	cout << "HepMCProduct non existent" << endl;
      }

      if(!ifGen){
	Handle<GenParticleCollection> genParticles; 
	try {
	  event.getByLabel ("genParticles", genParticles);
	  plotter->fillGen1(genParticles);
	  if (debug>0) cout << "Found genParticles" << endl;
	} catch (...) {
	  cout << "GenParticles non existent" << endl;
	}
      }

     Handle<SimTrackContainer> simTracks;
      try {
	event.getByLabel ("g4SimHits",simTracks);
	plotter->fillSim(simTracks);
	if(ifGen){
	  plotter->fillGenSim(evtMC,simTracks);
	}
      }
      catch (...) { 
	cout << "SimTracks not existent" << endl;
      }
    }
  }

  // On the first iteration we read the bank, otherwise we fetch the information from the muon tree
  // ----------------------------------------------------------------------------------------------
  if (loopCounter==0) {

    recMu1 = reco::Particle::LorentzVector(0,0,0,0);
    recMu2 = reco::Particle::LorentzVector(0,0,0,0);
    vector<reco::LeafCandidate> muons;
    if (theMuonType==1) { // GlobalMuons
      Handle<reco::MuonCollection> glbMuons;
      event.getByLabel (theMuonLabel, glbMuons);
      muons = fillMuonCollection(*glbMuons);
    }

    else if (theMuonType==2) { // StandaloneMuons
      Handle<reco::TrackCollection> saMuons;
      event.getByLabel (theMuonLabel, saMuons);
      muons = fillMuonCollection(*saMuons);
    }
    
    else if (theMuonType==3) { // Tracker tracks
      Handle<reco::TrackCollection> tracks;
      event.getByLabel (theMuonLabel, tracks);
      muons = fillMuonCollection(*tracks);
    }
    plotter->fillRec(muons);

  // Find the two muons from the resonance, and set ResFound bool
  // ------------------------------------------------------------
   pair <reco::Particle::LorentzVector, reco::Particle::LorentzVector> recMuFromBestRes = 
      MuScleFitUtils::findBestRecoRes (muons);
    if (MuScleFitUtils::ResFound) {
      if (debug>0) {
	cout <<setprecision(9)<< "Pt after findbestrecores: " << (recMuFromBestRes.first).Pt() << " " 
	     << (recMuFromBestRes.second).Pt() << endl;
	cout << "recMu1 = " << recMu1 << endl;
	cout << "recMu2 = " << recMu2 << endl;
      }
      recMu1 = recMuFromBestRes.first;
      recMu2 = recMuFromBestRes.second;
      if (debug>0) {
	cout << "after recMu1 = " << recMu1 << endl;
	cout << "after recMu2 = " << recMu2 << endl;
	cout << "mu1.pt = " << recMu1.Pt() << endl;
	cout << "mu2.pt = " << recMu2.Pt() << endl;
      }
      MuScleFitUtils::SavedPair.push_back (make_pair (recMu1, recMu2));
    } else {
      MuScleFitUtils::SavedPair.push_back (make_pair (lorentzVector(0.,0.,0.,0.), lorentzVector(0.,0.,0.,0.)));
    }

  } else {

    // On loops>0 the two muons are directly obtained from the SavedMuon array
    // -----------------------------------------------------------------------

    recMu1 = (MuScleFitUtils::SavedPair[iev].first);
    recMu2 = (MuScleFitUtils::SavedPair[iev].second);
    if (recMu1.Pt()>0 && recMu2.Pt()>0) {
      MuScleFitUtils::ResFound = true;
      if (debug>0) cout << "Ev = " << iev << ": found muons in tree with Pt = " 
			<< recMu1.Pt() << " " << recMu2.Pt() << endl;
    }
  }

  if (debug>0) cout << "About to start lik par correction and histo filling; ResFound is " 
		    << MuScleFitUtils::ResFound << endl;
  // If resonance found, do the hard work
  // ------------------------------------
  if (MuScleFitUtils::ResFound) {

    // Find weight and reference mass for this muon pair
    // -------------------------------------------------
    double weight = MuScleFitUtils::computeWeight ((recMu1+recMu2).mass());
    
    // Apply the correction (or bias) to the best 2 reconstructed muons
    // NNBB work in progress - try to establish how to deal with multiple correction
    // of the same scale if loop is made multiple times to fit resolution at step>3:
    // uncomment debug>0, take off loopCounter<3 condition below to restore.
    // -----------------------------------------------------------------------------  
    if (debug>0) {
      cout << "Loop #" << loopCounter << "Event #" << iev << ": before correction     Pt1 = " 
	   << recMu1.Pt() << " Pt2 = " << recMu2.Pt() << endl;
    }
    // If likelihood has been run already, we can correct and "unbias" muons with the latest fit results 
    // -------------------------------------------------------------------------------------------------
    if (loopCounter>0 && loopCounter<3) {
      recMu1 = (MuScleFitUtils::applyScale (recMu1, MuScleFitUtils::parvalue[loopCounter-1], -1));
      recMu2 = (MuScleFitUtils::applyScale (recMu2, MuScleFitUtils::parvalue[loopCounter-1],  1));
    }
    if (debug>0) {
      cout << "Loop #" << loopCounter << "Event #" << iev << ": after correction      Pt1 = " 
	   << recMu1.Pt() << " Pt2 = " << recMu2.Pt() << endl;
    } 

    reco::Particle::LorentzVector bestRecRes (recMu1+recMu2);

    //Fill histograms
    //------------------
    mapHisto["hRecBestMu"]->Fill(recMu1);
    if ((abs(recMu1.eta())<2.5) && (recMu1.pt()>2.5)) {
      mapHisto["hRecBestMu_Acc"]->Fill(recMu1);
    }
    mapHisto["hRecBestMu"]->Fill(recMu2);
    if ((abs(recMu2.eta())<2.5) && (recMu2.pt()>2.5)) {
      mapHisto["hRecBestMu_Acc"]->Fill(recMu2);
    }
    mapHisto["hDeltaRecBestMu"]->Fill(recMu1, recMu2);
    
    mapHisto["hRecBestRes"]->Fill(bestRecRes);
    if ((abs(recMu1.eta())<2.5) && (recMu1.pt()>2.5) && (abs(recMu2.eta())<2.5) &&  (recMu2.pt()>2.5)){
      mapHisto["hRecBestRes_Acc"]->Fill(bestRecRes);
      // Fill histogram of Res mass vs muon variable
      mapHisto["hRecBestResVSMu"]->Fill (recMu1, bestRecRes, -1);
      mapHisto["hRecBestResVSMu"]->Fill (recMu2, bestRecRes, +1);
    }

    //Compute pt resolution w.r.t generated and simulated muons
    //--------------------------------------------------------
    Handle<HepMCProduct> evtMC;
    try {
      event.getByLabel ("source", evtMC);
    } catch (...) { 
      cout << "HepMCProduct non existent" << endl;
    }

    Handle<SimTrackContainer> simTracks;
      try {
	event.getByLabel ("g4SimHits",simTracks);
	plotter->fillSim(simTracks);
      }
      catch (...) { 
	cout << "SimTracks not existent" << endl;
      } 

    pair <reco::Particle::LorentzVector, reco::Particle::LorentzVector> genMu = 
      MuScleFitUtils::findGenMuFromRes(evtMC);
   //first is always mu-, second is always mu+
   if(checkDeltaR(genMu.first,recMu1)){
       mapHisto["hResolPtGenVSMu"]->Fill(genMu.first,(-genMu.first.Pt()+recMu1.Pt())/genMu.first.Pt(),-1);
       mapHisto["hResolEtaGenVSMu"]->Fill(genMu.first,(-genMu.first.Eta()+recMu1.Eta()),-1);
       mapHisto["hResolPhiGenVSMu"]->Fill(genMu.first,(-genMu.first.Phi()+recMu1.Phi()),-1);
    }
   if(checkDeltaR(genMu.second,recMu2)){
       mapHisto["hResolPtGenVSMu"]->Fill(genMu.second,(-genMu.second.Pt()+recMu2.Pt())/genMu.second.Pt(),+1);
       mapHisto["hResolEtaGenVSMu"]->Fill(genMu.second,(-genMu.second.Eta()+recMu2.Eta()),+1);
       mapHisto["hResolPhiGenVSMu"]->Fill(genMu.second,(-genMu.second.Phi()+recMu2.Phi()),+1);
    }
    pair <reco::Particle::LorentzVector, reco::Particle::LorentzVector> simMu = 
      MuScleFitUtils::findSimMuFromRes(evtMC,simTracks);
   //first is always mu-, second is always mu+
   if(checkDeltaR(simMu.first,recMu1)){
       mapHisto["hResolPtSimVSMu"]->Fill(simMu.first,(-simMu.first.Pt()+recMu1.Pt())/simMu.first.Pt(),-1);
       mapHisto["hResolEtaSimVSMu"]->Fill(simMu.first,(-simMu.first.Eta()+recMu1.Eta()),-1);
       mapHisto["hResolPhiSimVSMu"]->Fill(simMu.first,(-simMu.first.Phi()+recMu1.Phi()),-1);
    }
   if(checkDeltaR(simMu.second,recMu2)){
       mapHisto["hResolPtSimVSMu"]->Fill(simMu.second,(-simMu.second.Pt()+recMu2.Pt())/simMu.first.Pt(),+1);
       mapHisto["hResolEtaSimVSMu"]->Fill(simMu.second,(-simMu.second.Eta()+recMu2.Eta()),+1);
       mapHisto["hResolPhiSimVSMu"]->Fill(simMu.second,(-simMu.second.Phi()+recMu2.Phi()),+1);
    }

    // Compute likelihood histograms
    // -----------------------------
    if (weight!=0.) {
      double massResol;
      double prob;
      double deltalike;
      if (loopCounter==0) {
	vector<double> initpar;
	for (int i=0; i<(int)(MuScleFitUtils::parResol.size()); i++) {
	  initpar.push_back(MuScleFitUtils::parResol[i]);
	}
	for (int i=0; i<(int)(MuScleFitUtils::parScale.size()); i++) {
	  initpar.push_back(MuScleFitUtils::parScale[i]);
	}
	for (int i=0; i<(int)(MuScleFitUtils::parBgr.size()); i++) {
	  initpar.push_back(MuScleFitUtils::parBgr[i]);
	}
	massResol = MuScleFitUtils::massResolution (recMu1, recMu2, initpar);
	prob      = MuScleFitUtils::massProb (bestRecRes.mass(), massResol, initpar);
      } else {
	massResol = MuScleFitUtils::massResolution (recMu1, recMu2, 
							  MuScleFitUtils::parvalue[loopCounter-1]);
	prob      = MuScleFitUtils::massProb (bestRecRes.mass(), 
						    massResol, MuScleFitUtils::parvalue[loopCounter-1]);
      }
      if (prob>0) { 
	deltalike = log(prob)*weight; // NB maximum likelihood --> deltalike is maximized
	mapHisto["hLikeVSMu"]->Fill (recMu1, deltalike);
	mapHisto["hLikeVSMu"]->Fill (recMu2, deltalike);
	mapHisto["hLikeVSMuMinus"]->Fill (recMu1, deltalike);
	mapHisto["hLikeVSMuPlus"]->Fill (recMu2, deltalike);
	mapHisto["hResolMassVSMu"]->Fill (recMu1, massResol,-1);
	mapHisto["hResolMassVSMu"]->Fill (recMu2, massResol,+1);

	Mass_P->Fill(bestRecRes.mass(), prob);
	Mass_fine_P->Fill(bestRecRes.mass(), prob);
      }
    }
    
    // Compute f, g for each variable
    // ------------------------------
    if (!MuScleFitUtils::speedup) {
      for (int i=0; i<6; i++) {
	if (bestRecRes.mass()>minResMass_hwindow[i] && bestRecRes.mass()<maxResMass_hwindow[i]) {
	  MuScleFitUtils::computeEstimator (recMu1, recMu2, bestRecRes.mass());
	}
      }
    }
    
  } // end if ResFound  

  // Fill the pair
  // -------------
  if (loopCounter>0) {
    if (debug>0) cout << "[MuScleFit]: filling the pair" << endl;
    MuScleFitUtils::SavedPair[iev] = make_pair (recMu1, recMu2);
  }
  
  iev++;
  return kContinue;
}

// Histogram filling
// -----------------
void MuScleFit::fillHistoMap() {
  //Reconstructed muon kinematics
  //-----------------------------
  mapHisto["hRecBestMu"]      = new HParticle ("hRecBestMu");
  mapHisto["hRecBestMu_Acc"]  = new HParticle ("hRecBestMu_Acc"); 
  mapHisto["hDeltaRecBestMu"]      = new HDelta ("hDeltaRecBestMu");

  mapHisto["hRecBestRes"]      = new HParticle   ("hRecBestRes");
  mapHisto["hRecBestRes_Acc"]  = new HParticle   ("hRecBestRes_Acc"); 
  mapHisto["hRecBestResVSMu"]  = new HMassVSPart ("hRecBestResVSMu");
  
  // Likelihood values VS muon variables
  // -------------------------------------
  mapHisto["hLikeVSMu"]      =  new HLikelihoodVSPart ("hLikeVSMu");
  mapHisto["hLikeVSMuMinus"] =  new HLikelihoodVSPart ("hLikeVSMuMinus");
  mapHisto["hLikeVSMuPlus"]  =  new HLikelihoodVSPart ("hLikeVSMuPlus");

  //Resolution VS muon kinematic
  //----------------------------
  mapHisto["hResolMassVSMu"] =  new HResolutionVSPart ("hResolMassVSMu");
  mapHisto["hResolPtGenVSMu"] =  new HResolutionVSPart ("hResolPtGenVSMu");
  mapHisto["hResolPtSimVSMu"] =  new HResolutionVSPart ("hResolPtSimVSMu");
  mapHisto["hResolEtaGenVSMu"] =  new HResolutionVSPart ("hResolEtaGenVSMu");
  mapHisto["hResolEtaSimVSMu"] =  new HResolutionVSPart ("hResolEtaSimVSMu");
  mapHisto["hResolPhiGenVSMu"] =  new HResolutionVSPart ("hResolPhiGenVSMu");
  mapHisto["hResolPhiSimVSMu"] =  new HResolutionVSPart ("hResolPhiSimVSMu");

  // Mass probability histograms
  // ---------------------------
  Mass_P = new TProfile ("Mass_P", "Mass probability", 4000, 0., 200., 0., 1.);
  Mass_fine_P = new TProfile ("Mass_fine_P", "Mass probability", 4000, 0., 20., 0., 1.);
}

bool MuScleFit::checkDeltaR(reco::Particle::LorentzVector& genMu, reco::Particle::LorentzVector& recMu){
  //first is always mu-, second is always mu+
  double deltaR = sqrt(MuScleFitUtils::deltaPhi(recMu.Phi(),genMu.Phi()) * MuScleFitUtils::deltaPhi(recMu.Phi(),genMu.Phi()) +
			 ((recMu.Eta()-genMu.Eta()) * (recMu.Eta()-genMu.Eta())));
  if(deltaR<0.01)
    return true;
  else
    cout<<"Reco muon "<<recMu<<" with eta "<<recMu.Eta()<<" and phi "<<recMu.Phi()<<endl
	<<" DOES NOT MATCH with generated muon from resonance: "<<endl
	<<genMu<<" with eta "<<genMu.Eta()<<" and phi "<<genMu.Phi()<<endl;
  return false;
}

void MuScleFit::clearHistoMap() {
  for (map<string, Histograms*>::const_iterator histo=mapHisto.begin(); 
       histo!=mapHisto.end(); histo++) {
    delete (*histo).second;
  }
}

void MuScleFit::writeHistoMap() {
  for (map<string, Histograms*>::const_iterator histo=mapHisto.begin(); 
       histo!=mapHisto.end(); histo++) {
    (*histo).second->Write();
  }
}
