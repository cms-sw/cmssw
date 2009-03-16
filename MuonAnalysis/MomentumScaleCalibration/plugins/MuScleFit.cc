//  \class MuScleFit
//  Fitter of momentum scale and resolution from resonance decays to muon track pairs
//
// <<<<<<< MuScleFit.cc
//  $Date: 2009/03/05 15:14:07 $
//  $Revision: 1.29 $
// =======
//  $Date: 2009/03/05 15:14:07 $
//  $Revision: 1.29 $
// >>>>>>> 1.25
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
//
//
//    Notes on additions, T.Dorigo 22/12/2008
//    ---------------------------------------
//
//  - File Probs_new_1000_CTEQ.root now contains a set of 24 additional two-dim histograms,
//    defining the probability distribution of Z boson decays as a function of measured mass
//    and expected sigma in 24 different bins of Z rapidity, extracted from CTEQ 6 PDF (at
//    Leading Order) from the convolution in the factorization integral. See programs CTEQ.cpp
//    and Fits.C.
//  - The probability for Z boson events now thus depends on the measured rapidity of the dimuon
//    system. All functions in file MuScleFitUtils.cc have been suitably changed.
//
// ----------------------------------------------------------------------------------

#include "MuScleFit.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Histograms.h"
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
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TFile.h"
#include "TTree.h"
#include "TMinuit.h"
#include <vector>

// To use callgrind for code profiling uncomment also the following define.
#define USE_CALLGRIND
#include "valgrind/callgrind.h"

#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"

// To read likelihood distributions from the database.
	//#include "CondFormats/RecoMuonObjects/interface/MuScleFitLikelihoodPdf.h"
	//#include "CondFormats/DataRecord/interface/MuScleFitLikelihoodPdfRcd.h"

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
  // Selection of fits according to loop
  MuScleFitUtils::doResolFit = pset.getParameter<vector<int> >("doResolFit");
  MuScleFitUtils::doScaleFit = pset.getParameter<vector<int> >("doScaleFit");
  MuScleFitUtils::doBackgroundFit = pset.getParameter<vector<int> >("doBackgroundFit");

  // Bias and smear types
  // --------------------
  int biasType = pset.getParameter<int>("BiasType");
  MuScleFitUtils::BiasType = biasType;
  // No error, the scale functions are used also for the bias
  MuScleFitUtils::biasFunction = scaleFunctionVecService( biasType );
  int smearType = pset.getParameter<int>("SmearType");
  MuScleFitUtils::SmearType = smearType;
  MuScleFitUtils::smearFunction = smearFunctionService( smearType );

  // Fit types
  // ---------
  int resolFitType = pset.getParameter<int>("ResolFitType");
  MuScleFitUtils::ResolFitType = resolFitType;
  MuScleFitUtils::resolutionFunction = resolutionFunctionService( resolFitType );
  MuScleFitUtils::resolutionFunctionForVec = resolutionFunctionVecService( resolFitType );
  int scaleType = pset.getParameter<int>("ScaleFitType");
  MuScleFitUtils::ScaleFitType = scaleType;
  MuScleFitUtils::scaleFunction = scaleFunctionService( scaleType );
  MuScleFitUtils::scaleFunctionForVec = scaleFunctionVecService( scaleType );
  int backgroundType = pset.getParameter<int>("BgrFitType");
  MuScleFitUtils::BgrFitType   = backgroundType;
  MuScleFitUtils::backgroundFunction = backgroundFunctionService( backgroundType );
  MuScleFitUtils::backgroundFunction->setLeftWindowFactor(MuScleFitUtils::leftWindowFactor);
  MuScleFitUtils::backgroundFunction->setRightWindowFactor(MuScleFitUtils::rightWindowFactor);

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

  // Read the Probs file from database. If false it searches the root file in
  // MuonAnalysis/MomentumScaleCalibration/test of the active release.
  // readPdfFromDB = pset.getParameter<bool>("readPdfFromDB");

  // Check for parameters consistency
  // it will abort in case of errors.
  checkParameters();

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

  // Read probability distributions from the database of a local root file.
  // These are 2-D PDFs containing a grid of 1000x1000 values of the
  // integral of Lorentz * Gaussian as a function of mass and resolution
  // of a given measurement, for each of the six considered di-muon resonances.
  // -------------------------------------------------
  // if( readPdfFromDB ) readProbabilityDistributions( eventSetup );
  // else
  readProbabilityDistributionsFromFile();

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
  // theFiles[iLoop]->cd();
  fillHistoMap(theFiles[iLoop], iLoop);

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
  theFiles[iLoop]->cd();
  Mass_P->Write();
  Mass_fine_P->Write();
  PtminvsY->Write();
  PtmaxvsY->Write();
  EtamuvsY->Write();
  Y->Write();
  MY->Write();
  MYP->Write();
  YL->Write();
  PL->Write();
  PTL->Write();
  GM->Write();
  SM->Write();
  GSM->Write();

  // Likelihood minimization to compute corrections
  // ----------------------------------------------
  theFiles[iLoop]->cd();
  MuScleFitUtils::minimizeLikelihood();

  // ATTENTION, this was put BEFORE the minimizeLikelihood. Check for problems.
  theFiles[iLoop]->Close();

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

#ifdef USE_CALLGRIND
  CALLGRIND_START_INSTRUMENTATION;
#endif

  // Update the services
  // -------------------
  theService->update (eventSetup);
  if (debug>0) {
    cout << "[MuScleFit-duringLoop]: loopCounter = " << loopCounter
	 << " Run: " << event.id().run() << " Event: " << event.id().event() << endl;
  }

  // Find and store in histograms the generated and simulated resonance and muons
  // ----------------------------------------------------------------------------
  Handle<HepMCProduct> evtMC;
  Handle<SimTrackContainer> simTracks;
  if (!MuScleFitUtils::speedup) { // NB we skip the simulation part if we are in a hurry
    bool ifGen=false;
    try {
      event.getByLabel ("source", evtMC);
      // Fill gen information only in the first loop
      if( loopCounter == 0 ) plotter->fillGen2(evtMC);
      ifGen=true;
    } catch (...) { 
      cout << "HepMCProduct non existent" << endl;
    }

    if(!ifGen){
      Handle<GenParticleCollection> genParticles; 
      try {
        event.getByLabel ("genParticles", genParticles);
        // Fill gen information only in the first loop
        if( loopCounter == 0 ) plotter->fillGen1(genParticles);
        if (debug>0) cout << "Found genParticles" << endl;
      } catch (...) {
        cout << "GenParticles non existent" << endl;
      }
    }

    try {
      event.getByLabel ("g4SimHits",simTracks);
      plotter->fillSim(simTracks);
      if(ifGen && loopCounter == 0){
        plotter->fillGenSim(evtMC,simTracks);
      }
    }
    catch (...) { 
      cout << "SimTracks not existent" << endl;
    }
  }

  // On the first iteration we read the bank, otherwise we fetch the information from the muon tree
  // ------------------------------------ Important Note --------------------------------------- //
  // The fillMuonCollection method applies any smearing or bias to the muons, so we NEVER use
  // unbiased muons.
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
    MuScleFitUtils::ResFound = false;
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
    if ( loopCounter>0 ) {
      if ( MuScleFitUtils::doScaleFit[loopCounter-1] ) {
        recMu1 = (MuScleFitUtils::applyScale (recMu1, MuScleFitUtils::parvalue[loopCounter-1], -1));
        recMu2 = (MuScleFitUtils::applyScale (recMu2, MuScleFitUtils::parvalue[loopCounter-1],  1));
      }
    }
    if (debug>0) {
      cout << "Loop #" << loopCounter << "Event #" << iev << ": after correction      Pt1 = " 
	   << recMu1.Pt() << " Pt2 = " << recMu2.Pt() << endl;
    }

    reco::Particle::LorentzVector bestRecRes (recMu1+recMu2);

    //Fill histograms
    //------------------
    if (recMu1.Pt()>recMu2.Pt()) {
      PtminvsY->Fill(recMu2.Pt(),bestRecRes.Rapidity());
      PtmaxvsY->Fill(recMu1.Pt(),bestRecRes.Rapidity());
      EtamuvsY->Fill(recMu1.Eta(),bestRecRes.Rapidity());
      EtamuvsY->Fill(recMu2.Eta(),bestRecRes.Rapidity());
    } else {
      PtmaxvsY->Fill(recMu2.Pt(),bestRecRes.Rapidity());
      PtminvsY->Fill(recMu1.Pt(),bestRecRes.Rapidity());
      EtamuvsY->Fill(recMu1.Eta(),bestRecRes.Rapidity());
      EtamuvsY->Fill(recMu2.Eta(),bestRecRes.Rapidity());
    }
    Y->Fill(fabs(bestRecRes.Rapidity()));
    MY->Fill(fabs(bestRecRes.Rapidity()),bestRecRes.mass());
    MYP->Fill(fabs(bestRecRes.Rapidity()),bestRecRes.mass());
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

    vector<double> * parval;
    vector<double> initpar;
    // Store a pointer to the vector of parameters of the last iteration, or the initial
    // parameters if this is the first iteration
    if (loopCounter==0) {
      initpar = MuScleFitUtils::parResol;
      initpar.insert( initpar.end(), MuScleFitUtils::parScale.begin(), MuScleFitUtils::parScale.end() );
      initpar.insert( initpar.end(), MuScleFitUtils::parBgr.begin(), MuScleFitUtils::parBgr.end() );
      parval = &initpar;
    } else {
      parval = &(MuScleFitUtils::parvalue[loopCounter-1]);
    }

    //Compute pt resolution w.r.t generated and simulated muons
    //--------------------------------------------------------
    pair <reco::Particle::LorentzVector, reco::Particle::LorentzVector> genMu;
    if( !MuScleFitUtils::speedup ) {
      genMu = MuScleFitUtils::findGenMuFromRes(evtMC);
      //first is always mu-, second is always mu+

      double genmass = (genMu.first+genMu.second).mass();
      GM->Fill(genmass);
      if(checkDeltaR(genMu.first,recMu1)){
        //       mapHisto["hResolPtGenVSMu"]->Fill(genMu.first,(-genMu.first.Pt()+recMu1.Pt())/genMu.first.Pt(),-1);
        //       mapHisto["hResolThetaGenVSMu"]->Fill(genMu.first,(-genMu.first.Theta()+recMu1.Theta()),-1);
        //       mapHisto["hResolCotgThetaGenVSMu"]->Fill(genMu.first,(-cos(genMu.first.Theta())/sin(genMu.first.Theta())
        //                                                             +cos(recMu1.Theta())/sin(recMu1.Theta())),-1);
        //       mapHisto["hResolEtaGenVSMu"]->Fill(genMu.first,(-genMu.first.Eta()+recMu1.Eta()),-1);
        //       mapHisto["hResolPhiGenVSMu"]->Fill(genMu.first,(-genMu.first.Phi()+recMu1.Phi()),-1);

        // Draw with respect to recMu quantities, not genMu ones
        mapHisto["hResolPtGenVSMu"]->Fill(recMu1,(-genMu.first.Pt()+recMu1.Pt())/genMu.first.Pt(),-1);
        mapHisto["hResolThetaGenVSMu"]->Fill(recMu1,(-genMu.first.Theta()+recMu1.Theta()),-1);
        mapHisto["hResolCotgThetaGenVSMu"]->Fill(recMu1,(-cos(genMu.first.Theta())/sin(genMu.first.Theta())
                                                         +cos(recMu1.Theta())/sin(recMu1.Theta())),-1);
        mapHisto["hResolEtaGenVSMu"]->Fill(recMu1,(-genMu.first.Eta()+recMu1.Eta()),-1);
        // mapHisto["hResolPhiGenVSMu"]->Fill(recMu1,(-genMu.first.Phi()+recMu1.Phi()),-1);
        mapHisto["hResolPhiGenVSMu"]->Fill(recMu1,MuScleFitUtils::deltaPhiNoFabs(recMu1.Phi(), genMu.first.Phi()),-1);
        // Fill also the resolution histogramsm using the resolution functions:
        // the parameters are those from the last iteration, as the muons up to this point have also the
        // corrections from the same iteration.
        // Need to use a different array (ForVec), containing functors able to operate on vector<double>
        mapHisto["hFunctionResolPt"]->Fill( recMu1, MuScleFitUtils::resolutionFunctionForVec->sigmaPt(recMu1.Pt(), recMu1.Eta(), *parval ), -1 );
        mapHisto["hFunctionResolCotgTheta"]->Fill( recMu1, MuScleFitUtils::resolutionFunctionForVec->sigmaCotgTh(recMu1.Pt(), recMu1.Eta(), *parval ), -1 );
        mapHisto["hFunctionResolPhi"]->Fill( recMu1, MuScleFitUtils::resolutionFunctionForVec->sigmaPhi(recMu1.Pt(), recMu1.Eta(), *parval ), -1 );
      }
      if(checkDeltaR(genMu.second,recMu2)){
        //       mapHisto["hResolPtGenVSMu"]->Fill(genMu.second,(-genMu.second.Pt()+recMu2.Pt())/genMu.second.Pt(),+1);
        //       mapHisto["hResolThetaGenVSMu"]->Fill(genMu.second,(-genMu.second.Theta()+recMu2.Theta()),+1);
        //       mapHisto["hResolCotgThetaGenVSMu"]->Fill(genMu.second,(-cos(genMu.second.Theta())/sin(genMu.second.Theta())
        // 							     +cos(recMu2.Theta())/sin(recMu2.Theta())),+1);
        //       mapHisto["hResolEtaGenVSMu"]->Fill(genMu.second,(-genMu.second.Eta()+recMu2.Eta()),+1);
        //       mapHisto["hResolPhiGenVSMu"]->Fill(genMu.second,(-genMu.second.Phi()+recMu2.Phi()),+1);

        mapHisto["hResolPtGenVSMu"]->Fill(recMu2,(-genMu.second.Pt()+recMu2.Pt())/genMu.second.Pt(),+1);
        mapHisto["hResolThetaGenVSMu"]->Fill(recMu2,(-genMu.second.Theta()+recMu2.Theta()),+1);
        mapHisto["hResolCotgThetaGenVSMu"]->Fill(recMu2,(-cos(genMu.second.Theta())/sin(genMu.second.Theta())
                                                         +cos(recMu2.Theta())/sin(recMu2.Theta())),+1);
        mapHisto["hResolEtaGenVSMu"]->Fill(recMu2,(-genMu.second.Eta()+recMu2.Eta()),+1);
        // mapHisto["hResolPhiGenVSMu"]->Fill(recMu2,(-genMu.second.Phi()+recMu2.Phi()),+1);
        mapHisto["hResolPhiGenVSMu"]->Fill(recMu2,MuScleFitUtils::deltaPhiNoFabs(recMu2.Phi(), genMu.second.Phi()),+1);
        // Fill also the resolution histogramsm using the resolution functions
        mapHisto["hFunctionResolPt"]->Fill( recMu2, MuScleFitUtils::resolutionFunctionForVec->sigmaPt(recMu2.Pt(), recMu2.Eta(), *parval ), +1 );
        mapHisto["hFunctionResolCotgTheta"]->Fill( recMu2, MuScleFitUtils::resolutionFunctionForVec->sigmaCotgTh(recMu2.Pt(), recMu2.Eta(), *parval ), +1 );
        mapHisto["hFunctionResolPhi"]->Fill( recMu2, MuScleFitUtils::resolutionFunctionForVec->sigmaPhi(recMu2.Pt(), recMu2.Eta(), *parval ), +1 );
      }
      pair <reco::Particle::LorentzVector, reco::Particle::LorentzVector> simMu = 
        MuScleFitUtils::findSimMuFromRes(evtMC,simTracks);
      //first is always mu-, second is always mu+
      double simmass = (simMu.first+simMu.second).mass();
      SM->Fill(simmass);
      GSM->Fill(genmass-simmass);

      if(checkDeltaR(simMu.first,recMu1)){
        mapHisto["hResolPtSimVSMu"]->Fill(simMu.first,(-simMu.first.Pt()+recMu1.Pt())/simMu.first.Pt(),-1);
        mapHisto["hResolThetaSimVSMu"]->Fill(simMu.first,(-simMu.first.Theta()+recMu1.Theta()),-1);
        mapHisto["hResolCotgThetaSimVSMu"]->Fill(simMu.first,(-cos(simMu.first.Theta())/sin(simMu.first.Theta())
                                                              +cos(recMu1.Theta())/sin(recMu1.Theta())),-1);
        mapHisto["hResolEtaSimVSMu"]->Fill(simMu.first,(-simMu.first.Eta()+recMu1.Eta()),-1);
        // mapHisto["hResolPhiSimVSMu"]->Fill(simMu.first,(-simMu.first.Phi()+recMu1.Phi()),-1);
        mapHisto["hResolPhiSimVSMu"]->Fill(simMu.first,MuScleFitUtils::deltaPhiNoFabs(recMu1.Phi(), simMu.first.Phi()),-1);
      }
      if(checkDeltaR(simMu.second,recMu2)){
        mapHisto["hResolPtSimVSMu"]->Fill(simMu.second,(-simMu.second.Pt()+recMu2.Pt())/simMu.second.Pt(),+1);
        mapHisto["hResolThetaSimVSMu"]->Fill(simMu.second,(-simMu.second.Theta()+recMu2.Theta()),+1);
        mapHisto["hResolCotgThetaSimVSMu"]->Fill(simMu.second,(-cos(simMu.second.Theta())/sin(simMu.second.Theta())
                                                               +cos(recMu2.Theta())/sin(recMu2.Theta())),+1);
        mapHisto["hResolEtaSimVSMu"]->Fill(simMu.second,(-simMu.second.Eta()+recMu2.Eta()),+1);
        // mapHisto["hResolPhiSimVSMu"]->Fill(simMu.second,(-simMu.second.Phi()+recMu2.Phi()),+1);
        mapHisto["hResolPhiSimVSMu"]->Fill(simMu.second,MuScleFitUtils::deltaPhiNoFabs(recMu2.Phi(), simMu.second.Phi()),+1);
      }
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
	prob      = MuScleFitUtils::massProb (bestRecRes.mass(), bestRecRes.Rapidity(), massResol, initpar);
      } else {
	massResol = MuScleFitUtils::massResolution (recMu1, recMu2, 
                                                    MuScleFitUtils::parvalue[loopCounter-1]);
	prob      = MuScleFitUtils::massProb (bestRecRes.mass(), bestRecRes.Rapidity(), 
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
	YL->Fill(bestRecRes.Rapidity(),deltalike);
	PL->Fill(bestRecRes.P(),deltalike);
	PTL->Fill(bestRecRes.Pt(),deltalike);

        double recoMass = (recMu1+recMu2).mass();
        if( !MuScleFitUtils::speedup ) {
          double genMass = (genMu.first + genMu.second).mass();
          // Fill the mass resolution (computed from MC), we use the covariance class to compute the variance
          if( genMass != 0 ) {
            // double diffMass = (recoMass - genMass)/genMass;
            double diffMass = recoMass - genMass;
            // Fill if for both muons
            double pt1 = recMu1.pt();
            double eta1 = recMu1.eta();
            double pt2 = recMu2.pt();
            double eta2 = recMu2.eta();
            // This is to avoid nan
            if( diffMass == diffMass ) {
              massResolutionVsPtEta_->Fill(pt1, eta1, diffMass, diffMass);
              massResolutionVsPtEta_->Fill(pt2, eta2, diffMass, diffMass);
            }
            else {
              cout << "Error, there is a nan: recoMass = " << recoMass << ", genMass = " << genMass << endl;
            }
          }
          // Fill with mass resolution from resolution function
          double massRes = MuScleFitUtils::massResolution(recMu1, recMu2, MuScleFitUtils::parResol);
          // The value given by massRes is already divided by the mass, since the derivative functions have mass at the denominator.
          mapHisto["hFunctionResolMass"]->Fill( recMu1, pow(massRes,2), -1 );
          mapHisto["hFunctionResolMass"]->Fill( recMu2, pow(massRes,2), +1 );
        }


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

#ifdef USE_CALLGRIND
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif

  return kContinue;
}

// Histogram filling
// -----------------
void MuScleFit::fillHistoMap(TFile* outputFile, unsigned int iLoop) {
  //Reconstructed muon kinematics
  //-----------------------------
  outputFile->cd();
  // If no Z is required, use a smaller mass range.
  double minMass = 0.;
  double maxMass = 200.;
  if( MuScleFitUtils::resfind[0] != 1 ) {
    maxMass = 30.;
  }
  mapHisto["hRecBestMu"]      = new HParticle ("hRecBestMu", minMass, maxMass);
  mapHisto["hRecBestMu_Acc"]  = new HParticle ("hRecBestMu_Acc", minMass, maxMass); 
  mapHisto["hDeltaRecBestMu"] = new HDelta ("hDeltaRecBestMu");

  mapHisto["hRecBestRes"]     = new HParticle   ("hRecBestRes", minMass, maxMass);
  mapHisto["hRecBestRes_Acc"] = new HParticle   ("hRecBestRes_Acc", minMass, maxMass); 
  // If not finding Z, use a smaller mass window
  vector<int>::const_iterator resFindIt = MuScleFitUtils::resfind.begin();
  mapHisto["hRecBestResVSMu"] = new HMassVSPart ("hRecBestResVSMu", minMass, maxMass);
  
  // Likelihood values VS muon variables
  // -------------------------------------
  mapHisto["hLikeVSMu"]       = new HLikelihoodVSPart ("hLikeVSMu");
  mapHisto["hLikeVSMuMinus"]  = new HLikelihoodVSPart ("hLikeVSMuMinus");
  mapHisto["hLikeVSMuPlus"]   = new HLikelihoodVSPart ("hLikeVSMuPlus");

  //Resolution VS muon kinematic
  //----------------------------
  mapHisto["hResolMassVSMu"]         = new HResolutionVSPart (outputFile, "hResolMassVSMu");
  mapHisto["hResolPtGenVSMu"]        = new HResolutionVSPart (outputFile, "hResolPtGenVSMu");
  mapHisto["hResolPtSimVSMu"]        = new HResolutionVSPart (outputFile, "hResolPtSimVSMu");
  mapHisto["hResolEtaGenVSMu"]       = new HResolutionVSPart (outputFile, "hResolEtaGenVSMu");
  mapHisto["hResolEtaSimVSMu"]       = new HResolutionVSPart (outputFile, "hResolEtaSimVSMu");
  mapHisto["hResolThetaGenVSMu"]     = new HResolutionVSPart (outputFile, "hResolThetaGenVSMu");
  mapHisto["hResolThetaSimVSMu"]     = new HResolutionVSPart (outputFile, "hResolThetaSimVSMu");
  mapHisto["hResolCotgThetaGenVSMu"] = new HResolutionVSPart (outputFile, "hResolCotgThetaGenVSMu");
  mapHisto["hResolCotgThetaSimVSMu"] = new HResolutionVSPart (outputFile, "hResolCotgThetaSimVSMu");
  mapHisto["hResolPhiGenVSMu"]       = new HResolutionVSPart (outputFile, "hResolPhiGenVSMu");
  mapHisto["hResolPhiSimVSMu"]       = new HResolutionVSPart (outputFile, "hResolPhiSimVSMu");

  // Resolutions from resolution functions
  // -------------------------------------
  mapHisto["hFunctionResolPt"]        = new HFunctionResolution (outputFile, "hFunctionResolPt");
  mapHisto["hFunctionResolCotgTheta"] = new HFunctionResolution (outputFile, "hFunctionResolCotgTheta");
  mapHisto["hFunctionResolPhi"]       = new HFunctionResolution (outputFile, "hFunctionResolPhi");

  // Mass probability histograms
  // ---------------------------
  Mass_P = new TProfile ("Mass_P", "Mass probability", 4000, 0., 200., 0., 1.);
  Mass_fine_P = new TProfile ("Mass_fine_P", "Mass probability", 4000, 0., 20., 0., 1.);
  PtminvsY = new TH2D ("PtminvsY","PtminvsY",120, 0., 120., 120, 0., 6.);
  PtmaxvsY = new TH2D ("PtmaxvsY","PtmaxvsY",120, 0., 120., 120, 0., 6.);
  EtamuvsY = new TH2D ("EtamuvsY","EtamuvsY",120, 0., 3., 120, 0., 6.);
  Y = new TH1D ("Y","Y", 100, 0., 5. );
  MY = new TH2D ("MY","MY",100, 0., 5., 100, 0., 200.);
  MYP = new TProfile ("MYP","MYP",100, 0., 5., 0.,200.);
  YL = new TProfile("YL","YL", 40, -4., 4., -10000000.,10000000.);
  PL = new TProfile("PL","PL", 40, 0., 200., -10000000.,10000000.);
  PTL = new TProfile("PTL","PTL", 40, 0., 100., -10000000., 10000000.);
  GM = new TH1D ("GM","GM", 120, 61., 121.);
  SM = new TH1D ("SM","SM", 120, 61., 121.);
  GSM = new TH1D("GSM","GSM", 100, -2.5, 2.5);

  double ptMax = 40.;
  // Mass resolution vs (pt, eta) of the muons from MC
  massResolutionVsPtEta_ = new HCovarianceVSxy ( "Mass", "Mass", 100, 0., ptMax, 60, -3, 3, outputFile->mkdir("MassCovariance") );
  // Mass resolution vs (pt, eta) from resolution function
  mapHisto["hFunctionResolMass"] = new HFunctionResolution (outputFile, "hFunctionResolMass", ptMax);
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
  massResolutionVsPtEta_->Clear();
  delete massResolutionVsPtEta_;
}

void MuScleFit::writeHistoMap() {
  for (map<string, Histograms*>::const_iterator histo=mapHisto.begin(); 
       histo!=mapHisto.end(); histo++) {
    (*histo).second->Write();
  }
  // theFiles[iLoop]->cd();
  massResolutionVsPtEta_->Write();
}

// Simple method to check parameters consistency. It aborts the job if the parameters
// are not consistent.
void MuScleFit::checkParameters() {

  // Fits selection dimension check
  if( MuScleFitUtils::doResolFit.size() != maxLoopNumber ) {
    cout << "[MuScleFit-Constructor]: wrong size of resolution fits selector = " << MuScleFitUtils::doResolFit.size() << endl;
    cout << "it must have as many values as the number of loops, which is = " << maxLoopNumber << endl;
    abort();
  }
  if( MuScleFitUtils::doScaleFit.size() != maxLoopNumber ) {
    cout << "[MuScleFit-Constructor]: wrong size of scale fits selector = " << MuScleFitUtils::doScaleFit.size() << endl;
    cout << "it must have as many values as the number of loops, which is = " << maxLoopNumber << endl;
    abort();
  }
  if( MuScleFitUtils::doBackgroundFit.size() != maxLoopNumber ) {
    cout << "[MuScleFit-Constructor]: wrong size of background fits selector = " << MuScleFitUtils::doBackgroundFit.size() << endl;
    cout << "it must have as many values as the number of loops, which is = " << maxLoopNumber << endl;
    abort();
  }

  // Bias parameters: dimension check
  // --------------------------------
  if ((MuScleFitUtils::BiasType==1  && MuScleFitUtils::parBias.size()!=2) || // linear in pt
      (MuScleFitUtils::BiasType==2  && MuScleFitUtils::parBias.size()!=2) || // linear in |eta|
      (MuScleFitUtils::BiasType==3  && MuScleFitUtils::parBias.size()!=2) || // sinusoidal in phi
      (MuScleFitUtils::BiasType==4  && MuScleFitUtils::parBias.size()!=3) || // linear in pt and |eta|
      (MuScleFitUtils::BiasType==5  && MuScleFitUtils::parBias.size()!=3) || // linear in pt and sinusoidal in phi
      (MuScleFitUtils::BiasType==6  && MuScleFitUtils::parBias.size()!=3) || // linear in |eta| and sinusoidal in phi
      (MuScleFitUtils::BiasType==7  && MuScleFitUtils::parBias.size()!=4) || // linear in pt and |eta| and sinusoidal in phi
      (MuScleFitUtils::BiasType==8  && MuScleFitUtils::parBias.size()!=4) || // linear in pt and parabolic in |eta|
      (MuScleFitUtils::BiasType==9  && MuScleFitUtils::parBias.size()!=2) || // exponential in pt
      (MuScleFitUtils::BiasType==10 && MuScleFitUtils::parBias.size()!=3) || // parabolic in pt
      (MuScleFitUtils::BiasType==11 && MuScleFitUtils::parBias.size()!=4) || // linear in pt and sin in phi with chg
      (MuScleFitUtils::BiasType==12 && MuScleFitUtils::parBias.size()!=6) || // linear in pt and para in plus sin in phi with chg
      (MuScleFitUtils::BiasType==13 && MuScleFitUtils::parBias.size()!=8) || // linear in pt and para in plus sin in phi with chg
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
      (MuScleFitUtils::ResolFitType==6 && MuScleFitUtils::parResol.size()!=15) ||
      (MuScleFitUtils::ResolFitType==7 && MuScleFitUtils::parResol.size()!=12) ||
      (MuScleFitUtils::ResolFitType==8 && MuScleFitUtils::parResol.size()!=11) ||
      MuScleFitUtils::ResolFitType<1 || MuScleFitUtils::ResolFitType>8) {
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
      (MuScleFitUtils::ScaleFitType==6  && MuScleFitUtils::parScale.size()!=3) || // linear in |eta| and sinusoidal in phi
      (MuScleFitUtils::ScaleFitType==7  && MuScleFitUtils::parScale.size()!=4) || // linear in pt and |eta| and sinusoidal in phi
      (MuScleFitUtils::ScaleFitType==8  && MuScleFitUtils::parScale.size()!=4) || // linear in pt and parabolic in |eta|
      (MuScleFitUtils::ScaleFitType==9  && MuScleFitUtils::parScale.size()!=2) || // exponential in pt
      (MuScleFitUtils::ScaleFitType==10 && MuScleFitUtils::parScale.size()!=3) || // parabolic in pt
      (MuScleFitUtils::ScaleFitType==11 && MuScleFitUtils::parScale.size()!=4) || // linear in pt and sin in phi w/ chg
      (MuScleFitUtils::ScaleFitType==12 && MuScleFitUtils::parScale.size()!=6) || // linear in pt and para in eta plus sin in phi with chg
      (MuScleFitUtils::ScaleFitType==13 && MuScleFitUtils::parScale.size()!=8) || // linear in pt and para in eta plus sin in phi with chg
      (MuScleFitUtils::ScaleFitType==14 && MuScleFitUtils::parScale.size()!=8) || // parabolic in pt and 6th grade polynomial in |eta|
      MuScleFitUtils::ScaleFitType<1 || MuScleFitUtils::ScaleFitType>15) {
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
}

void MuScleFit::readProbabilityDistributionsFromFile()
{
  TH2D * GLZ[24];
  TH2D * GL[6];
  TFile * ProbsFile;
  if ( theMuonType!=2 ) {
    //edm::FileInPath file("MuonAnalysis/MomentumScaleCalibration/test/Probs_new_1000_CTEQ.root");
    edm::FileInPath file("MuonAnalysis/MomentumScaleCalibration/test/Probs_new_Horace_CTEQ_1000.root");
    ProbsFile = new TFile (file.fullPath().c_str()); // NNBB need to reset this if MuScleFitUtils::nbins changes
    // ProbsFile = new TFile ("Probs_new_1000_CTEQ.root"); // NNBB need to reset this if MuScleFitUtils::nbins changes
    //cout << "[MuScleFit-Constructor]: Reading TH2D probabilities from Probs_new_1000_CTEQ.root file" << endl;
    cout << "[MuScleFit-Constructor]: Reading TH2D probabilities from Probs_new_Horace_CTEQ_1000.root file" << endl;
  } else {
    edm::FileInPath fileSM("MuonAnalysis/MomentumScaleCalibration/test/Probs_SM_1000.root");
    ProbsFile = new TFile (fileSM.fullPath().c_str()); // NNBB need to reset this if MuScleFitUtils::nbins changes
    // ProbsFile = new TFile ("Probs_SM_1000.root"); // NNBB need to reset this if MuScleFitUtils::nbins changes
    cout << "[MuScleFit-Constructor]: Reading TH2D probabilities from Probs_new_SM_1000.root file" << endl;
  }
  ProbsFile->cd();
  for ( int i=0; i<24; i++ ) {
    char nameh[6];
    sprintf (nameh,"GLZ%d",i);
    GLZ[i] = dynamic_cast<TH2D*>(ProbsFile->Get(nameh)); 
  }
  GL[0] = dynamic_cast<TH2D*> (ProbsFile->Get("GL0"));
  GL[1] = dynamic_cast<TH2D*> (ProbsFile->Get("GL1"));
  GL[2] = dynamic_cast<TH2D*> (ProbsFile->Get("GL2"));
  GL[3] = dynamic_cast<TH2D*> (ProbsFile->Get("GL3"));
  GL[4] = dynamic_cast<TH2D*> (ProbsFile->Get("GL4"));
  GL[5] = dynamic_cast<TH2D*> (ProbsFile->Get("GL5"));

  // Extract normalization for mass slice in Y bins of Z
  // ---------------------------------------------------
  for (int iY=0; iY<24; iY++) {
    int nBinsX = GLZ[iY]->GetNbinsX();
    int nBinsY = GLZ[iY]->GetNbinsY();
    if( nBinsX != MuScleFitUtils::nbins+1 || nBinsY != MuScleFitUtils::nbins+1 ) {
      cout << "Error: for histogram \"" << GLZ[iY]->GetName() << "\" bins are not " << MuScleFitUtils::nbins << endl; 
      cout<< "nBinsX = " << nBinsX << ", nBinsY = " << nBinsY << endl;
      exit(1);
    }
    for (int iy=0; iy<=MuScleFitUtils::nbins; iy++) {
      MuScleFitUtils::GLZNorm[iY][iy] = 0.;
      for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
        MuScleFitUtils::GLZValue[iY][ix][iy] = GLZ[iY]->GetBinContent (ix+1, iy+1);
        MuScleFitUtils::GLZNorm[iY][iy] += MuScleFitUtils::GLZValue[iY][ix][iy];
      }
      if (debug>2) cout << "GLZValue[" << iY << "][500][" << iy << "] = " 
                        << MuScleFitUtils::GLZValue[iY][500][iy] 
                        << " GLZNorm[" << iY << "][" << iy << "] = " 
                        << MuScleFitUtils::GLZNorm[iY][iy] << endl;
    }
  }
  // Extract normalization for each mass slice
  // -----------------------------------------
  for (int ires=0; ires<6; ires++) {
    int nBinsX = GL[ires]->GetNbinsX();
    int nBinsY = GL[ires]->GetNbinsY();
    if( nBinsX != MuScleFitUtils::nbins+1 || nBinsY != MuScleFitUtils::nbins+1 ) {
      cout << "Error: for histogram \"" << GLZ[ires]->GetName() << "\" bins are not " << MuScleFitUtils::nbins << endl; 
      cout<< "nBinsX = " << nBinsX << ", nBinsY = " << nBinsY << endl;
      exit(1);
    }
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
}

// void MuScleFit::readProbabilityDistributions( const edm::EventSetup & eventSetup )
// {

//   edm::ESHandle<MuScleFitLikelihoodPdf> likelihoodPdf;
//   eventSetup.get<MuScleFitLikelihoodPdfRcd>().get(likelihoodPdf);
//   string smSuffix = "";

//   // Should read different histograms in the two cases
//   if ( theMuonType == 2 ) {
//     smSuffix = "SM";
//     cout << "Error: Not yet implemented..." << endl;
//     exit(1);
//   }

//   edm::LogInfo("MuScleFit") << "[MuScleFit::readProbabilityDistributions] End Reading MuScleFitLikelihoodPdfRcd" << endl;
//   vector<PhysicsTools::Calibration::HistogramD2D>::const_iterator histo = likelihoodPdf->histograms.begin();
//   vector<string>::const_iterator name = likelihoodPdf->names.begin();
//   vector<int>::const_iterator xBins = likelihoodPdf->xBins.begin();
//   vector<int>::const_iterator yBins = likelihoodPdf->yBins.begin();
//   int ires = 0;
//   int iY = 0;
//   for( ; histo != likelihoodPdf->histograms.end(); ++histo, ++name, ++xBins, ++yBins ) {
//     int nBinsX = *xBins;
//     int nBinsY = *yBins;
//     if( nBinsX != MuScleFitUtils::nbins+1 || nBinsY != MuScleFitUtils::nbins+1 ) {
//       cout << "Error: for histogram \"" << *name << "\" bins are not " << MuScleFitUtils::nbins << endl; 
//       cout<< "nBinsX = " << nBinsX << ", nBinsY = " << nBinsY << endl;
//       exit(1);
//     }

//     // cout << "name = " << *name << endl;

//     // To separate the Z histograms from the other resonances we use tha names.
//     if( name->find("GLZ") != string::npos ) {
//       // ATTENTION: they are expected to be ordered

//       // cout << "For iY = " << iY << " the histogram is \"" << *name << "\"" << endl;

//       // Extract normalization for mass slice in Y bins of Z
//       // ---------------------------------------------------
//       for(int iy=1; iy<=nBinsY; iy++){
//         MuScleFitUtils::GLZNorm[iY][iy] = 0.;
//         for(int ix=1; ix<=nBinsX; ix++){
//           MuScleFitUtils::GLZValue[iY][ix][iy] = histo->binContent (ix+1, iy+1);
//           MuScleFitUtils::GLZNorm[iY][iy] += MuScleFitUtils::GLZValue[iY][ix][iy];
//         }
//         if (debug>2) cout << "GLZValue[" << iY << "][500][" << iy << "] = " 
//                           << MuScleFitUtils::GLZValue[iY][500][iy] 
//                           << " GLZNorm[" << iY << "][" << iy << "] = " 
//                           << MuScleFitUtils::GLZNorm[iY][iy] << endl;
//       }
//       // increase the histogram counter
//       ++iY;
//     }
//     else {
//       // ATTENTION: they are expected to be ordered

//       // Extract normalization for each mass slice
//       // -----------------------------------------

//       // cout << "For ires = " << ires << " the histogram is \"" << *name << "\"" << endl;

//       // The histograms are filled like the root TH2D from which they are taken,
//       // meaning that bin = 0 is the underflow and nBins+1 is the overflow.
//       // We start from 1 and loop up to the last bin, excluding under/overflow.
//       for(int iy=1; iy<=nBinsY; iy++){
//         MuScleFitUtils::GLNorm[ires][iy] = 0.;
//         for(int ix=1; ix<=nBinsX; ix++){
//           MuScleFitUtils::GLValue[ires][ix][iy] = histo->binContent (ix+1, iy+1);
//           MuScleFitUtils::GLNorm[ires][iy] += MuScleFitUtils::GLValue[ires][ix][iy];
//         }
//         if (debug>2) cout << "GLValue[" << ires << "][500][" << iy << "] = " 
//                           << MuScleFitUtils::GLValue[ires][500][iy] 
//                           << " GLNorm[" << ires << "][" << iy << "] = " 
//                           << MuScleFitUtils::GLNorm[ires][iy] << endl;
//       }
//       // increase the histogram counter
//       ++ires;
//     }
//   }
// }
