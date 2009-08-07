//  \class MuScleFit
//  Fitter of momentum scale and resolution from resonance decays to muon track pairs
//
//  $Date: 2009/07/30 13:06:28 $
//  $Revision: 1.50 $
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
//    Modifications by M.De Mattia 13/3/2009
//    --------------------------------------
//  - The histograms map was moved to a base class (MuScleFitBase) from which this one inherits.
//
// ---------------------------------------------------------------------------------------------

#include "MuScleFit.h"
// #include "MuonAnalysis/MomentumScaleCalibration/interface/Histograms.h"
// #include "MuScleFitUtils.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/MuScleFitPlotter.h"
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
//#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TFile.h"
#include "TTree.h"
#include "TMinuit.h"
#include <vector>

// To use callgrind for code profiling uncomment also the following define.
// #define USE_CALLGRIND

#ifdef USE_CALLGRIND
#include "valgrind/callgrind.h"
#endif

#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"

// To read likelihood distributions from the database.
	//#include "CondFormats/RecoMuonObjects/interface/MuScleFitLikelihoodPdf.h"
	//#include "CondFormats/DataRecord/interface/MuScleFitLikelihoodPdfRcd.h"

using namespace std;
using namespace edm;
using namespace reco; // For AODSIM MC objects

// Constructor
// -----------
MuScleFit::MuScleFit( const ParameterSet& pset ) : MuScleFitBase( pset ), totalEvents_(0)
{
  MuScleFitUtils::debug = debug_;
  if (debug_>0) cout << "[MuScleFit]: Constructor" << endl;

  // Service parameters
  // ------------------
  ParameterSet serviceParameters = pset.getParameter<ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);

  if ((theMuonType_<1 || theMuonType_>5) && theMuonType_!=10) {
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

  MuScleFitUtils::backgroundHandler = new BackgroundHandler( pset.getParameter<vector<int> >("BgrFitType"),
                                                             pset.getParameter<vector<double> >("LeftWindowFactor"),
                                                             pset.getParameter<vector<double> >("RightWindowFactor"),
                                                             MuScleFitUtils::ResMass );

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

  // Option to skip simTracks comparison
  compareToSimTracks_ = pset.getParameter<bool>("compareToSimTracks");
  simTracksCollection_ = pset.getUntrackedParameter<InputTag>("SimTracksCollection", InputTag("g4SimHits"));

  // This must be set to true if using events generated with Sherpa
  MuScleFitUtils::sherpa_ = pset.getUntrackedParameter<bool>("Sherpa", false);

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
    TF1 G("G", "[0]*exp(-0.5*pow(x,2))", -5., 5.);
    double norm = 1/sqrt(2*TMath::Pi());
    G.SetParameter (0,norm);
    for (int i=0; i<10000; i++) {
      for (int j=0; j<7; j++) {
	MuScleFitUtils::x[j][i] = G.GetRandom();
      }
    }
  }
  MuScleFitUtils::goodmuon = 0;

  // Initialize ResMaxSigma And ResHalfWidth - 0 = global, 1 = SM, 2 = tracker
  // -------------------------------------------------------------------------
//   MuScleFitUtils::ResMaxSigma[0] = 50.;
//   MuScleFitUtils::ResMaxSigma[1] = 5.;
//   MuScleFitUtils::ResMaxSigma[2] = 5.;
//   MuScleFitUtils::ResMaxSigma[3] = 5.;
//   MuScleFitUtils::ResMaxSigma[4] = 2.;
//   MuScleFitUtils::ResMaxSigma[5] = 2.;
//   MuScleFitUtils::ResHalfWidth[0] = 20.;
//   MuScleFitUtils::ResHalfWidth[1] = 0.5;
//   MuScleFitUtils::ResHalfWidth[2] = 0.5;
//   MuScleFitUtils::ResHalfWidth[3] = 0.5;
//   MuScleFitUtils::ResHalfWidth[4] = 0.2;
//   MuScleFitUtils::ResHalfWidth[5] = 0.2;
  MuScleFitUtils::massWindowHalfWidth[0][0] = 20.;
  MuScleFitUtils::massWindowHalfWidth[1][0] = 0.5;
  MuScleFitUtils::massWindowHalfWidth[2][0] = 0.5;
  MuScleFitUtils::massWindowHalfWidth[3][0] = 0.5;
  MuScleFitUtils::massWindowHalfWidth[4][0] = 0.2;
  MuScleFitUtils::massWindowHalfWidth[5][0] = 0.2;
  MuScleFitUtils::massWindowHalfWidth[0][1] = 50.;
  MuScleFitUtils::massWindowHalfWidth[1][1] = 2.5;
  MuScleFitUtils::massWindowHalfWidth[2][1] = 2.5;
  MuScleFitUtils::massWindowHalfWidth[3][1] = 2.5;
  MuScleFitUtils::massWindowHalfWidth[4][1] = 1.5;
  MuScleFitUtils::massWindowHalfWidth[5][1] = 1.5;
  MuScleFitUtils::massWindowHalfWidth[0][2] = 20.;
  MuScleFitUtils::massWindowHalfWidth[1][2] = 0.5;
  MuScleFitUtils::massWindowHalfWidth[2][2] = 0.5;
  MuScleFitUtils::massWindowHalfWidth[3][2] = 0.5;
  MuScleFitUtils::massWindowHalfWidth[4][2] = 0.2;
  MuScleFitUtils::massWindowHalfWidth[5][2] = 0.2;

  MuScleFitUtils::MuonType = theMuonType_-1;
  if( MuScleFitUtils::MuonType < 3 ) {
    MuScleFitUtils::MuonTypeForCheckMassWindow = theMuonType_-1;
  }
  else {
    MuScleFitUtils::MuonTypeForCheckMassWindow = 2;
  }
}

// Destructor
// ----------
MuScleFit::~MuScleFit () {
  if (debug_>0) cout << "[MuScleFit]: Destructor" << endl;
  cout << "Total number of analyzed events = " << totalEvents_ << endl;
  plotter->writeHistoMap();
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
  if(maxLoopNumber>1)
    readProbabilityDistributionsFromFile();

  if (debug_>0) cout << "[MuScleFit]: beginOfJob" << endl;
  
  // Create the root file
  // --------------------
  for (unsigned int i=0; i<(maxLoopNumber); i++) {
    char buffer [2]; // FIXME: the max number of loop has to be < 10, (otherwise not enough char)
    sprintf (buffer, "%d_", i);
    string rootFileName = buffer + theRootFileName_;
    theFiles_.push_back (new TFile(rootFileName.c_str(), "RECREATE"));
  }
  if (debug_>0) cout << "[MuScleFit]: Root file created" << endl;

  plotter = new MuScleFitPlotter(theGenInfoRootFileName_);
  plotter->debug = debug_; 
}

// End of job method
// -----------------              
void MuScleFit::endOfJob () {
  if (debug_>0) cout << "[MuScleFit]: endOfJob" << endl;
  //delete plotter;
}

// New loop
// --------
void MuScleFit::startingNewLoop (unsigned int iLoop) {

  if (debug_>0) cout << "[MuScleFit]: Starting loop # " << iLoop << endl;

  // Number of muons used 
  // --------------------
  MuScleFitUtils::goodmuon = 0;

  // Counters for problem cout-ing
  // -----------------------------
  MuScleFitUtils::counter_resprob = 0;

  // Create the root file
  // --------------------
  fillHistoMap(theFiles_[iLoop], iLoop);

  loopCounter = iLoop;
  MuScleFitUtils::loopCounter = loopCounter;

  iev = 0;
  MuScleFitUtils::iev_ = 0;

  MuScleFitUtils::cleanEstimator();
}

// End of loop routine
// -------------------
edm::EDLooper::Status MuScleFit::endOfLoop (const edm::EventSetup& eventSetup, unsigned int iLoop) {

  cout << "Ending loop # " << iLoop << endl;

  // Write the histos to file
  // ------------------------
  // theFiles_[iLoop]->cd();
  writeHistoMap(iLoop);

  // Likelihood minimization to compute corrections
  // ----------------------------------------------
  theFiles_[iLoop]->cd();
  MuScleFitUtils::minimizeLikelihood();

  // ATTENTION, this was put BEFORE the minimizeLikelihood. Check for problems.
  theFiles_[iLoop]->Close();
  // ATTENTION: Check that this delete does not give any problem
  delete theFiles_[iLoop];

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
  // theService->update (eventSetup);
  if (debug_>0) {
    cout << "[MuScleFit-duringLoop]: loopCounter = " << loopCounter
	 << " Run: " << event.id().run() << " Event: " << event.id().event() << endl;
  }

  // Find and store in histograms the generated and simulated resonance and muons
  // ----------------------------------------------------------------------------
  Handle<HepMCProduct> evtMC;
  Handle<SimTrackContainer> simTracks;
  Handle<GenParticleCollection> genParticles; 
  if (!MuScleFitUtils::speedup) { // NB we skip the simulation part if we are in a hurry
    ifHepMC=false;
    ifGenPart=false;
    try {
      event.getByLabel ("source", evtMC);
      // Fill gen information only in the first loop
      if( loopCounter == 0 ) plotter->fillGen2(evtMC);
      ifHepMC=true;
    } catch (...) { 
      cout << "HepMCProduct non existent" << endl;
    }
    
    if(!ifHepMC){
      try {
        event.getByLabel ("genParticles", genParticles);
        // Fill gen information only in the first loop
        if( loopCounter == 0 ) plotter->fillGen1(genParticles);
	ifGenPart=true;
        if (debug_>0) cout << "Found genParticles" << endl;
      } catch (...) {
        cout << "GenParticles non existent" << endl;
      }
    }

    if( compareToSimTracks_ ) {
      try {
        event.getByLabel (simTracksCollection_, simTracks);
        plotter->fillSim(simTracks);
        if(ifHepMC && loopCounter == 0){
          plotter->fillGenSim(evtMC,simTracks);
        }
      }
      catch (...) {
        cout << "SimTracks not existent" << endl;
      }
    }
  }

  // On the first iteration we read the bank, otherwise we fetch the information from the muon tree
  // ------------------------------------ Important Note --------------------------------------- //
  // The fillMuonCollection method applies any smearing or bias to the muons, so we NEVER use
  // unbiased muons.
  // ----------------------------------------------------------------------------------------------
  if (loopCounter==0) {

    ++totalEvents_;

    recMu1 = reco::Particle::LorentzVector(0,0,0,0);
    recMu2 = reco::Particle::LorentzVector(0,0,0,0);
    vector<reco::LeafCandidate> muons;
    if (theMuonType_<4 || theMuonType_==10) { // Muons (glb,sta,trk)
      Handle<reco::MuonCollection> allMuons;
      event.getByLabel (theMuonLabel_, allMuons);
      vector<reco::Track> tracks;

      for (vector<reco::Muon>::const_iterator muon = allMuons->begin(); muon != allMuons->end(); ++muon){
	// cout<<"muon "<<muon->isGlobalMuon()<<muon->isStandAloneMuon()<<muon->isTrackerMuon()<<endl;
	//NNBB: one muon can be of many kinds at once but with the theMuonType_ we are sure
	// to avoid double counting of the same muon
	if(muon->isGlobalMuon() && theMuonType_==1)
	  tracks.push_back(*(muon->globalTrack()));
	else if(muon->isGlobalMuon() && theMuonType_==10) //particular case!!
	  tracks.push_back(*(muon->innerTrack()));
	else if(muon->isStandAloneMuon() && theMuonType_==2)
	  tracks.push_back(*(muon->outerTrack()));
	else if(muon->isTrackerMuon() && theMuonType_==3)
	  tracks.push_back(*(muon->innerTrack()));
      }
      muons = fillMuonCollection(tracks);
    }

    else if(theMuonType_==4){  //CaloMuons
      Handle<reco::CaloMuonCollection> caloMuons;
      event.getByLabel (theMuonLabel_, caloMuons);
      vector<reco::Track> tracks;
      for (vector<reco::CaloMuon>::const_iterator muon = caloMuons->begin(); muon != caloMuons->end(); ++muon){
	tracks.push_back(*(muon->track()));  
      }
      muons = fillMuonCollection(tracks);
    }

    else if (theMuonType_==5) { // Inner tracker tracks
      Handle<reco::TrackCollection> tracks;
      event.getByLabel (theMuonLabel_, tracks);
      muons = fillMuonCollection(*tracks);
    }
    
    plotter->fillRec(muons);
    // Compare reco with mc
    // if( loopCounter == 0 && !MuScleFitUtils::speedup ) plotter->fillRecoVsGen(muons, evtMC);

    // Find the two muons from the resonance, and set ResFound bool
    // ------------------------------------------------------------
    pair <reco::Particle::LorentzVector, reco::Particle::LorentzVector> recMuFromBestRes = 
      MuScleFitUtils::findBestRecoRes (muons);
    if (MuScleFitUtils::ResFound) {
      if (debug_>0) {
	cout <<setprecision(9)<< "Pt after findbestrecores: " << (recMuFromBestRes.first).Pt() << " " 
	     << (recMuFromBestRes.second).Pt() << endl;
	cout << "recMu1 = " << recMu1 << endl;
	cout << "recMu2 = " << recMu2 << endl;
      }
      recMu1 = recMuFromBestRes.first;
      recMu2 = recMuFromBestRes.second;
      if (debug_>0) {
	cout << "after recMu1 = " << recMu1 << endl;
	cout << "after recMu2 = " << recMu2 << endl;
	cout << "mu1.pt = " << recMu1.Pt() << endl;
	cout << "mu2.pt = " << recMu2.Pt() << endl;
      }
      MuScleFitUtils::SavedPair.push_back( make_pair (recMu1, recMu2) );
    } else {
      MuScleFitUtils::SavedPair.push_back( make_pair (lorentzVector(0.,0.,0.,0.), lorentzVector(0.,0.,0.,0.)) );
    }

  } else {

    // On loops>0 the two muons are directly obtained from the SavedMuon array
    // -----------------------------------------------------------------------
    MuScleFitUtils::ResFound = false;
    recMu1 = (MuScleFitUtils::SavedPair[iev].first);
    recMu2 = (MuScleFitUtils::SavedPair[iev].second);
    if (recMu1.Pt()>0 && recMu2.Pt()>0) {
      MuScleFitUtils::ResFound = true;
      if (debug_>0) cout << "Ev = " << iev << ": found muons in tree with Pt = " 
			<< recMu1.Pt() << " " << recMu2.Pt() << endl;
    }
  }

  if (debug_>0) cout << "About to start lik par correction and histo filling; ResFound is " 
		    << MuScleFitUtils::ResFound << endl;
  // If resonance found, do the hard work
  // ------------------------------------
  if (MuScleFitUtils::ResFound) {

    // Find weight and reference mass for this muon pair
    // -------------------------------------------------
    double weight = MuScleFitUtils::computeWeight( (recMu1+recMu2).mass(), iev );
    if (debug_>0) {
      cout << "Loop #" << loopCounter << "Event #" << iev << ": before correction     Pt1 = " 
	   << recMu1.Pt() << " Pt2 = " << recMu2.Pt() << endl;
    }
    // For successive iterations, correct the muons only if the previous iteration was a scale fit.
    // --------------------------------------------------------------------------------------------
    if ( loopCounter>0 ) {
      if ( MuScleFitUtils::doScaleFit[loopCounter-1] ) {
        recMu1 = (MuScleFitUtils::applyScale (recMu1, MuScleFitUtils::parvalue[loopCounter-1], -1));
        recMu2 = (MuScleFitUtils::applyScale (recMu2, MuScleFitUtils::parvalue[loopCounter-1],  1));
      }
    }
    if (debug_>0) {
      cout << "Loop #" << loopCounter << "Event #" << iev << ": after correction      Pt1 = " 
	   << recMu1.Pt() << " Pt2 = " << recMu2.Pt() << endl;
    }

    reco::Particle::LorentzVector bestRecRes (recMu1+recMu2);

    //Fill histograms
    //------------------
    mapHisto_["hRecBestMu"]->Fill(recMu1);
    mapHisto_["hRecBestMuVSEta"]->Fill(recMu1);
    mapHisto_["hRecBestMu"]->Fill(recMu2);
    mapHisto_["hRecBestMuVSEta"]->Fill(recMu2);
    mapHisto_["hDeltaRecBestMu"]->Fill(recMu1, recMu2);
    // Reconstructed resonance
    mapHisto_["hRecBestRes"]->Fill(bestRecRes, weight);
    mapHisto_["hRecBestResAllEvents"]->Fill(bestRecRes, 1.);
    // Fill histogram of Res mass vs muon variable
    mapHisto_["hRecBestResVSMu"]->Fill (recMu1, bestRecRes, -1);
    mapHisto_["hRecBestResVSMu"]->Fill (recMu2, bestRecRes, +1);
    mapHisto_["hRecBestResVSRes"]->Fill (bestRecRes, bestRecRes, +1);

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
      if(ifHepMC)
	genMu = MuScleFitUtils::findGenMuFromRes(evtMC);
      else if(ifGenPart)
	genMu = MuScleFitUtils::findGenMuFromRes(genParticles);
      else
	cout<<"ERROR "<<"non generation info and speedup true!!!!!!!!!!!!"<<endl;
      //first is always mu-, second is always mu+

      // double genmass = (genMu.first+genMu.second).mass();
      if(checkDeltaR(genMu.first,recMu1)) {
        fillComparisonHistograms( genMu.first, recMu1, "Gen", -1 );
        // Fill also the resolution histogramsm using the resolution functions:
        // the parameters are those from the last iteration, as the muons up to this point have also the
        // corrections from the same iteration.
        // Need to use a different array (ForVec), containing functors able to operate on vector<double>
        mapHisto_["hFunctionResolPt"]->Fill( recMu1, MuScleFitUtils::resolutionFunctionForVec->sigmaPt(recMu1.Pt(), recMu1.Eta(), *parval ), -1 );
        mapHisto_["hFunctionResolCotgTheta"]->Fill( recMu1, MuScleFitUtils::resolutionFunctionForVec->sigmaCotgTh(recMu1.Pt(), recMu1.Eta(), *parval ), -1 );
        mapHisto_["hFunctionResolPhi"]->Fill( recMu1, MuScleFitUtils::resolutionFunctionForVec->sigmaPhi(recMu1.Pt(), recMu1.Eta(), *parval ), -1 );
      }
      if(checkDeltaR(genMu.second,recMu2)){
        fillComparisonHistograms( genMu.second, recMu2, "Gen", +1 );
        // Fill also the resolution histogramsm using the resolution functions
        mapHisto_["hFunctionResolPt"]->Fill( recMu2, MuScleFitUtils::resolutionFunctionForVec->sigmaPt(recMu2.Pt(), recMu2.Eta(), *parval ), +1 );
        mapHisto_["hFunctionResolCotgTheta"]->Fill( recMu2, MuScleFitUtils::resolutionFunctionForVec->sigmaCotgTh(recMu2.Pt(), recMu2.Eta(), *parval ), +1 );
        mapHisto_["hFunctionResolPhi"]->Fill( recMu2, MuScleFitUtils::resolutionFunctionForVec->sigmaPhi(recMu2.Pt(), recMu2.Eta(), *parval ), +1 );
      }
      if( compareToSimTracks_ ) {
        pair <reco::Particle::LorentzVector, reco::Particle::LorentzVector> simMu = 
          MuScleFitUtils::findSimMuFromRes(evtMC,simTracks);
        //first is always mu-, second is always mu+
        if(checkDeltaR(simMu.first,recMu1)){
          fillComparisonHistograms( simMu.first, recMu1, "Sim", -1 );
        }
        if(checkDeltaR(simMu.second,recMu2)){
          fillComparisonHistograms( simMu.second, recMu2, "Sim", +1 );
        }
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
	mapHisto_["hLikeVSMu"]->Fill (recMu1, deltalike);
	mapHisto_["hLikeVSMu"]->Fill (recMu2, deltalike);
	mapHisto_["hLikeVSMuMinus"]->Fill (recMu1, deltalike);
	mapHisto_["hLikeVSMuPlus"]->Fill (recMu2, deltalike);
	mapHisto_["hResolMassVSMu"]->Fill (recMu1, massResol, -1);
	mapHisto_["hResolMassVSMu"]->Fill (recMu2, massResol, +1);

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
              mapHisto_["hMassResolutionVsPtEta"]->Fill(pt1, eta1, diffMass, diffMass);
              mapHisto_["hMassResolutionVsPtEta"]->Fill(pt2, eta2, diffMass, diffMass);
            }
            else {
              cout << "Error, there is a nan: recoMass = " << recoMass << ", genMass = " << genMass << endl;
            }
          }
          // Fill with mass resolution from resolution function
          double massRes = MuScleFitUtils::massResolution(recMu1, recMu2, MuScleFitUtils::parResol);
          // The value given by massRes is already divided by the mass, since the derivative functions have mass at the denominator.
          mapHisto_["hFunctionResolMass"]->Fill( recMu1, pow(massRes,2), -1 );
          mapHisto_["hFunctionResolMass"]->Fill( recMu2, pow(massRes,2), +1 );
        }

        mapHisto_["hMass_P"]->Fill(bestRecRes.mass(), prob);
        mapHisto_["hMass_fine_P"]->Fill(bestRecRes.mass(), prob);
      }
    }

    // Compute f, g for each variable
    // ------------------------------
    if (!MuScleFitUtils::speedup) {
      for( int i=0; i<6; ++i ) {
	if( bestRecRes.mass() > minResMass_hwindow[i] && bestRecRes.mass() < maxResMass_hwindow[i] ) {
	  MuScleFitUtils::computeEstimator( recMu1, recMu2, bestRecRes.mass() );
	}
      }
    }
  } // end if ResFound  

  // Fill the pair
  // -------------
  if (loopCounter>0) {
    if (debug_>0) cout << "[MuScleFit]: filling the pair" << endl;
    MuScleFitUtils::SavedPair[iev] = make_pair (recMu1, recMu2);
  }

  iev++;
  MuScleFitUtils::iev_++;

#ifdef USE_CALLGRIND
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif

  return kContinue;
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

void MuScleFit::fillComparisonHistograms( const reco::Particle::LorentzVector & genMu, const reco::Particle::LorentzVector & recMu, const string & inputName, const int charge ) {
  string name(inputName + "VSMu");
  mapHisto_["hResolPt"+name]->Fill(recMu, (-genMu.Pt()+recMu.Pt())/genMu.Pt(), charge);
  mapHisto_["hResolTheta"+name]->Fill(recMu, (-genMu.Theta()+recMu.Theta()), charge);
  mapHisto_["hResolCotgTheta"+name]->Fill(recMu,(-cos(genMu.Theta())/sin(genMu.Theta())
                                                 +cos(recMu.Theta())/sin(recMu.Theta())), charge);
  mapHisto_["hResolEta"+name]->Fill(recMu, (-genMu.Eta()+recMu.Eta()),charge);
  mapHisto_["hResolPhi"+name]->Fill(recMu, MuScleFitUtils::deltaPhiNoFabs(recMu.Phi(), genMu.Phi()), charge);

  mapHisto_["hPtRecoVsPt"+inputName]->Fill(genMu.Pt(), recMu.Pt());
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
      (MuScleFitUtils::ResolFitType==8 && MuScleFitUtils::parResol.size()!=12) ||
      (MuScleFitUtils::ResolFitType==9 && MuScleFitUtils::parResol.size()!=31) ||
      (MuScleFitUtils::ResolFitType==10 && MuScleFitUtils::parResol.size()!=21) ||
      MuScleFitUtils::ResolFitType<1 || MuScleFitUtils::ResolFitType>10) {
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
      (MuScleFitUtils::ScaleFitType==14 && MuScleFitUtils::parScale.size()!=10) || // cubic in pt and 6th grade polynomial in |eta|
      (MuScleFitUtils::ScaleFitType==15 && MuScleFitUtils::parScale.size()!=5) || // two different slopes in pt and parabolic in |eta|
      MuScleFitUtils::ScaleFitType<1 || MuScleFitUtils::ScaleFitType>16) {
    cout << "[MuScleFit-Constructor]: Wrong fit type or number of parameters: aborting!" << endl;
    abort();
  }
//   // Bgr fit parameters: dimension check
//   // -----------------------------------
//   if ((MuScleFitUtils::BgrFitType==1 && MuScleFitUtils::parBgr.size()!=1) ||
//       (MuScleFitUtils::BgrFitType==2 && MuScleFitUtils::parBgr.size()!=2) ||
//       (MuScleFitUtils::BgrFitType==3 && MuScleFitUtils::parBgr.size()!=3) ||
//       MuScleFitUtils::BgrFitType<1 || MuScleFitUtils::BgrFitType>3) {
//     cout << "[MuScleFit-Constructor]: Wrong Bgr fit type or number of parameters: aborting!" << endl;
//     abort();
//   }

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


