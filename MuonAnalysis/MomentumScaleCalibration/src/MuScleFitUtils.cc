/** See header file for a class description 
 *
 *  $Date: 2010/01/04 12:02:27 $
 *  $Revision: 1.24 $
 *  \author S. Bolognesi - INFN Torino / T. Dorigo, M. De Mattia - INFN Padova
 */
// Some notes:
// - M(Z) after simulation needs to be extracted as a function of |y_Z| in order to be
//   a better reference point for calibration. In fact, the variation of PDF with y_Z
//   in production is sizable <---- need to check though.
// - ResHalfWidth needs to be optimized - this depends on the level of background.
// - Background parametrization still to be worked on, so far only a constant (type=1, and
//   parameter 2 fixed to 0) works.
// - weights have to be assigned to dimuon mass values in regions where different resonances
//   overlap, and one has to decide which resonance mass to assign the event to - this until
//   we implement in the fitter a sum of probabilities of an event to belong to different
//   resonances. The weight has to depend on the mass and has relative cross sections of
//   Y(1S), 2S, 3S as parameters. Some overlap is also expected in the J/psi-Psi(2S) region
//   when reconstructing masses with Standalone muons.
//
//   MODS 7/7/08 TD:
//   - changed parametrization of resolution in Pt: from sigma_pt = a*Pt + b*|eta| to 
//                                                       sigma_pt = (a*Pt + b*|eta|)*Pt
//                                                  which is more correct (I hope)
//   - changed parametrization of resolution in cotgth: from sigma_cotgth = f(eta) to f(cotgth)
// --------------------------------------------------------------------------------------------

#include "MuonAnalysis/MomentumScaleCalibration/interface/MuScleFitUtils.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TH2F.h"
#include "TF1.h"
#include "TF2.h"
#include <iostream>
#include <fstream>
#include <memory> // to use the std::auto_ptr

// Includes the definitions of all the bias and scale functions
// These functions are selected in the constructor according
// to the input parameters.
#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"

// To use callgrind for code profiling uncomment also the following define.
//#define USE_CALLGRIND
#ifdef USE_CALLGRIND
#include "valgrind/callgrind.h"
#endif

using namespace std;
using namespace edm;
using namespace reco;

// Lorenzian Peak function    
// -----------------------
Double_t lorentzianPeak (Double_t *x, Double_t *par) {
  return (0.5*par[0]*par[1]/TMath::Pi()) /
  TMath::Max(1.e-10,(x[0]-par[2])*(x[0]-par[2]) + .25*par[1]*par[1]);
}

// Gaussian function    
// -----------------
Double_t Gaussian (Double_t *x, Double_t *par) {
  return par[0]*exp(-0.5*((x[0]-par[1])/par[2])*((x[0]-par[1])/par[2])); 
}

// Array with number of parameters in the fitting functions
// (not currently in use)
// --------------------------------------------------------
//const int nparsResol[2] = {6, 4};
//const int nparsScale[13] = {2, 2, 2, 3, 3, 3, 4, 4, 2, 3, 4, 6, 8};
//const int nparsBgr[3] = {1, 2, 3};

// Quantities used for h-value computation
// ---------------------------------------
double mzsum;
double isum;
double f[11][100];
double g[11][100];

// Lorentzian convoluted with a gaussian:
// --------------------------------------
TF1 * GL = new TF1 ("GL", 
		    "0.5/3.1415926*[0]/(pow(x-[1],2)+pow(0.5*[0],2))*exp(-0.5*pow((x-[2])/[3],2))/([3]*sqrt(6.283185))", 
		    0, 1000);

TF2 * GL2= new TF2 ("GL2", 
  "0.5/3.1415926*[0]/(pow(x-[1],2)+pow(0.5*[0],2))*exp(-0.5*pow((x-y)/[2],2))/([2]*sqrt(6.283185))", 
  0, 200, 0, 200);

// // Lorentzian convoluted with a gaussian over a linear background:
// // ---------------------------------------------------------------
// TF1 * GLBL = new TF1 ("GLBL", 
//   "0.5/3.1415926*[0]/(pow(x-[1],2)+pow(0.5*[0],2))*exp(-0.5*pow((x-[2])/[3],2))/([3]*sqrt(6.283185))+[4]+[5]*x", 
//   0, 1000);

// // Lorentzian convoluted with a gaussian over an exponential background:
// // ---------------------------------------------------------------
// TF1 * GLBE = new TF1 ("GLBE", 
//   "0.5/3.1415926*[0]/(pow(x-[1],2)+pow(0.5*[0],2))*exp(-0.5*pow((x-[2])/[3],2))/([3]*sqrt(6.283185))+exp([4]+[5]*x)", 
//   0, 1000);

vector<int> MuScleFitUtils::doResolFit;
vector<int> MuScleFitUtils::doScaleFit;
vector<int> MuScleFitUtils::doBackgroundFit;

int MuScleFitUtils::minuitLoop_ = 0;
TH1D* MuScleFitUtils::likelihoodInLoop_ = 0;
TH1D* MuScleFitUtils::signalProb_ = 0;
TH1D* MuScleFitUtils::backgroundProb_ = 0;

const int MuScleFitUtils::totalResNum = 6;

int MuScleFitUtils::SmearType = 0;
smearFunctionBase * MuScleFitUtils::smearFunction = 0;
int MuScleFitUtils::BiasType  = 0;
// No error, we take functions from the same group for bias and scale.
scaleFunctionBase<vector<double> > * MuScleFitUtils::biasFunction = 0;
int MuScleFitUtils::ResolFitType = 0;
resolutionFunctionBase<double *> * MuScleFitUtils::resolutionFunction = 0;
resolutionFunctionBase<vector<double> > * MuScleFitUtils::resolutionFunctionForVec = 0;
int MuScleFitUtils::ScaleFitType = 0;
scaleFunctionBase<double*> * MuScleFitUtils::scaleFunction = 0;
scaleFunctionBase<vector<double> > * MuScleFitUtils::scaleFunctionForVec = 0;
int MuScleFitUtils::BgrFitType   = 0;
// const int MuScleFitUtils::backgroundFunctionsRegions = 3;
// backgroundFunctionBase * MuScleFitUtils::backgroundFunctionForRegion[MuScleFitUtils::backgroundFunctionsRegions];
// backgroundFunctionBase * MuScleFitUtils::backgroundFunction[MuScleFitUtils::totalResNum];
BackgroundHandler * MuScleFitUtils::backgroundHandler;
vector<double> MuScleFitUtils::parBias;
vector<double> MuScleFitUtils::parSmear;
vector<double> MuScleFitUtils::parResol;
vector<double> MuScleFitUtils::parScale;
vector<double> MuScleFitUtils::parBgr;
vector<int> MuScleFitUtils::parResolFix;
vector<int> MuScleFitUtils::parScaleFix;
vector<int> MuScleFitUtils::parBgrFix;
vector<int> MuScleFitUtils::parResolOrder;
vector<int> MuScleFitUtils::parScaleOrder;
vector<int> MuScleFitUtils::parBgrOrder;

vector<int> MuScleFitUtils::resfind;
int MuScleFitUtils::debug = 0;

bool MuScleFitUtils::ResFound = false;
int MuScleFitUtils::goodmuon = 0;
int MuScleFitUtils::counter_resprob = 0;

vector<vector<double> > MuScleFitUtils::parvalue;

int MuScleFitUtils::FitStrategy = 1; // Strategy in likelihood fit (1 or 2)
bool MuScleFitUtils::speedup = false; // Whether to cut corners (no sim study, fewer histos)

vector<pair<lorentzVector,lorentzVector> > MuScleFitUtils::SavedPair; // Pairs of reconstructed muons making resonances
vector<pair<lorentzVector,lorentzVector> > MuScleFitUtils::genPair; // Pairs of generated muons making resonances
vector<pair<lorentzVector,lorentzVector> > MuScleFitUtils::simPair; // Pairs of simulated muons making resonances

// Smearing parameters
// -------------------
double MuScleFitUtils::x[][10000];

// Probability matrices and normalization values
// ---------------------------------------------
int MuScleFitUtils::nbins = 1000;
double MuScleFitUtils::GLZValue[][1001][1001];
double MuScleFitUtils::GLZNorm[][1001];
double MuScleFitUtils::GLValue[][1001][1001];
double MuScleFitUtils::GLNorm[][1001];
double MuScleFitUtils::ResMaxSigma[];

// Masses and widths from PDG 2006, half widths to be revised
// NB in particular, halfwidths have to be made a function of muonType
// -------------------------------------------------------------------
const double MuScleFitUtils::mMu2 = 0.011163612;
const double MuScleFitUtils::muMass = 0.105658;
double MuScleFitUtils::ResHalfWidth[];
double MuScleFitUtils::massWindowHalfWidth[][3];
int MuScleFitUtils::MuonType;
int MuScleFitUtils::MuonTypeForCheckMassWindow;

double MuScleFitUtils::ResGamma[] = {2.4952, 0.000020, 0.000032, 0.000054, 0.000317, 0.0000932 };
double MuScleFitUtils::ResMass[] = {91.1876, 10.3552, 10.0233, 9.4603, 3.68609, 3.0969};
double MuScleFitUtils::ResMassForBackground[] = { MuScleFitUtils::ResMass[0],
                                                  (MuScleFitUtils::ResMass[1]+MuScleFitUtils::ResMass[2]+MuScleFitUtils::ResMass[3])/3,
                                                  (MuScleFitUtils::ResMass[4] + MuScleFitUtils::ResMass[5])/2 };
unsigned int MuScleFitUtils::loopCounter = 5;

const unsigned int MuScleFitUtils::motherPdgIdArray[] = {23, 200553, 100553, 553, 100443, 443};

// double MuScleFitUtils::leftWindowFactor = 1.;
// double MuScleFitUtils::rightWindowFactor = 1.;

// double MuScleFitUtils::internalLeftWindowFactor = 1.;
// double MuScleFitUtils::internalRightWindowFactor = 1.;

// int MuScleFitUtils::backgroundWindowEvents_ = 0;
// int MuScleFitUtils::resonanceWindowEvents_ = 0;

// double MuScleFitUtils::oldEventsOutInRatio_ = 0.;

bool MuScleFitUtils::scaleFitNotDone_ = true;

bool MuScleFitUtils::sherpa_ = false;

bool MuScleFitUtils::rapidityBinsForZ_ = true;

double MuScleFitUtils::minMuonPt_ = 0.;
double MuScleFitUtils::maxMuonPt_ = 100000000.;
double MuScleFitUtils::minMuonEtaFirstRange_ = -6.;
double MuScleFitUtils::maxMuonEtaFirstRange_ = 6.;
double MuScleFitUtils::minMuonEtaSecondRange_ = -100.;
double MuScleFitUtils::maxMuonEtaSecondRange_ = 100.;

bool MuScleFitUtils::debugMassResol_;
MuScleFitUtils::massResolComponentsStruct MuScleFitUtils::massResolComponents;

TMinuit * MuScleFitUtils::rminPtr_ = 0;

int MuScleFitUtils::iev_ = 0;
///////////////////////////////////////////////////////////////////////////////////////////////

// Find the best simulated resonance from a vector of simulated muons (SimTracks) 
// and return its decay muons
// ------------------------------------------------------------------------------
pair<SimTrack,SimTrack> MuScleFitUtils::findBestSimuRes (const vector<SimTrack>& simMuons) {

  pair<SimTrack, SimTrack> simMuFromBestRes;
  double maxprob = -0.1;

  // Double loop on muons
  // --------------------
  for (vector<SimTrack>::const_iterator simMu1=simMuons.begin(); simMu1!=simMuons.end(); simMu1++) {
    for (vector<SimTrack>::const_iterator simMu2=simMu1+1; simMu2!=simMuons.end(); simMu2++) {
      if (((*simMu1).charge()*(*simMu2).charge())>0) {
	continue; // this also gets rid of simMu1==simMu2...
      }
      // Choose the best resonance using its mass. Check Z, Y(3S,2S,1S), Psi(2S), J/Psi in order
      // ---------------------------------------------------------------------------------------
      double mcomb = ((*simMu1).momentum()+(*simMu2).momentum()).mass();
      double Y = ((*simMu1).momentum()+(*simMu2).momentum()).Rapidity();
      for (int ires=0; ires<6; ires++) {
	if (resfind[ires]>0) {
	  double prob = massProb (mcomb, Y, ires, 0.);
	  if (prob>maxprob) {
	    simMuFromBestRes.first = (*simMu1);
	    simMuFromBestRes.second = (*simMu2);
	    maxprob = prob;
	  }
	} 
      }
    }
  }

  // Return most likely combination of muons making a resonance
  // ----------------------------------------------------------
  return simMuFromBestRes;
}

// Find the best reconstructed resonance from a collection of reconstructed muons 
// (MuonCollection) and return its decay muons
// ------------------------------------------------------------------------------
pair<lorentzVector,lorentzVector> MuScleFitUtils::findBestRecoRes( const vector<reco::LeafCandidate>& muons ){
  // NB this routine returns the resonance, but it also sets the ResFound flag, which 
  // is used in MuScleFit to decide whether to use the event or not.
  // --------------------------------------------------------------------------------
  if (debug>0) cout << "In findBestRecoRes" << endl;
  ResFound = false;
  pair <lorentzVector, lorentzVector> recMuFromBestRes; 


  // Very simple method taking the two highest Pt muons in the event
  // ---------------------------------------------------------------
//   lorentzVector * firstMuon = 0;
//   lorentzVector * secondMuon = 0;
//   double firstMuonPt = 0;
//   double secondMuonPt = 0;

//   struct byPt
//   {
//     bool operator() ( const reco::LeafCandidate & muon1, const reco::LeafCandidate & muon2 ) {
//       return muon1.pt() > muon2.pt();
//     }
//   };

//   if( muons.size() > 1 ) {
//     vector<reco::LeafCandidate> muonsCopy( muons );
//     sort( muonsCopy, byPt );
//   }
//   else {
//     cout << "WARNING: Event has less than two muons" << endl;
//   }

    // Choose the best resonance using its mass probability
    // ----------------------------------------------------
  double maxprob = -0.1; 
  double minDeltaMass = 999999;
  pair<reco::LeafCandidate,reco::LeafCandidate> bestMassMuons;
  for (vector<reco::LeafCandidate>::const_iterator Muon1=muons.begin(); Muon1!=muons.end(); ++Muon1) {  
    for (vector<reco::LeafCandidate>::const_iterator Muon2=Muon1+1; Muon2!=muons.end(); ++Muon2) { 
      if (((*Muon1).charge()*(*Muon2).charge())>0) {
	continue; // This also gets rid of Muon1==Muon2...
      }
      // // Accept combinations only if both muons have |eta|<maxMuonEta_ and pt>minMuonPt_
      // // -------------------------------------------------------------------------------
      // if( (*Muon1).p4().Pt() > minMuonPt_ && (*Muon2).p4().Pt() > minMuonPt_ &&
      //     fabs((*Muon1).p4().Eta()) < maxMuonEta_ && fabs((*Muon2).p4().Eta()) < maxMuonEta_ ) {

      // To allow the selection of ranges at negative and positive eta independently we define two
      // ranges of eta: (minMuonEtaFirstRange_, maxMuonEtaFirstRange_) and (minMuonEtaSecondRange_, maxMuonEtaSecondRange_).
      // If the interval selected is simmetric, one only needs to specify the first range. The second has
      // default values that accept all muons (minMuonEtaSecondRange_ = -100., maxMuonEtaSecondRange_ = 100.).
      double pt1 = (*Muon1).p4().Pt();
      double pt2 = (*Muon2).p4().Pt();
      double eta1 = (*Muon1).p4().Eta();
      double eta2 = (*Muon2).p4().Eta();
      // cout << "eta1 = " << eta1 << ", eta2 = " << eta2 << ", minMuonEtaFirstRange_ = " << minMuonEtaFirstRange_ << ", maxMuonEtaFirstRange_ = " << maxMuonEtaFirstRange_ << ", minMuonEtaSecondRange_ = " << minMuonEtaSecondRange_ << ", maxMuonEtaSecondRange_ = " << maxMuonEtaSecondRange_ << endl;
      // bool pt1Cut = (pt1 > minMuonPt_ && pt1 < maxMuonPt_);
      // bool pt2Cut = (pt2 > minMuonPt_ && pt2 < maxMuonPt_);
      // bool eta1CutFirstRange = (eta1 > minMuonEtaFirstRange_ && eta1 < maxMuonEtaFirstRange_);
      // bool eta2CutFirstRange = (eta2 > minMuonEtaFirstRange_ && eta2 < maxMuonEtaFirstRange_);
      // bool eta1CutSecondRange = (eta1 > minMuonEtaSecondRange_ && eta1 < maxMuonEtaSecondRange_);
      // bool eta2CutSecondRange = (eta2 > minMuonEtaSecondRange_ && eta2 < maxMuonEtaSecondRange_);
      // if( pt1Cut && pt2Cut && eta1CutFirstRange && eta2CutFirstRange && eta1CutSecondRange && eta2CutSecondRange ) {
      if( pt1 > minMuonPt_ && pt1 < maxMuonPt_ &&
          pt2 > minMuonPt_ && pt2 < maxMuonPt_ &&
          eta1 > minMuonEtaFirstRange_ && eta1 < maxMuonEtaFirstRange_ &&
          eta2 > minMuonEtaFirstRange_ && eta2 < maxMuonEtaFirstRange_ &&
          eta1 > minMuonEtaSecondRange_ && eta1 < maxMuonEtaSecondRange_ &&
          eta2 > minMuonEtaSecondRange_ && eta2 < maxMuonEtaSecondRange_ ) {
        double mcomb = ((*Muon1).p4()+(*Muon2).p4()).mass();
        double Y = ((*Muon1).p4()+(*Muon2).p4()).Eta();
        if (debug>1) {
          cout<<"muon1 "<<(*Muon1).p4().Px()<<", "<<(*Muon1).p4().Py()<<", "<<(*Muon1).p4().Pz()<<", "<<(*Muon1).p4().E()<<endl;
          cout<<"muon2 "<<(*Muon2).p4().Px()<<", "<<(*Muon2).p4().Py()<<", "<<(*Muon2).p4().Pz()<<", "<<(*Muon2).p4().E()<<endl;
          cout<<"mcomb "<<mcomb<<endl;}
        double massResol = massResolution ((*Muon1).p4(), (*Muon2).p4(), parResol);
        double prob;
        for( int ires=0; ires<6; ires++ ) {
          if( resfind[ires]>0 ) {
            prob = massProb( mcomb, Y, ires, massResol );
            if( prob>maxprob ) {
              if( (*Muon1).charge()<0 ) { // store first the mu minus and then the mu plus
                recMuFromBestRes.first = (*Muon1).p4();
                recMuFromBestRes.second = (*Muon2).p4();
              } else {
                recMuFromBestRes.first = (*Muon2).p4();
                recMuFromBestRes.second = (*Muon1).p4();
              }
              ResFound = true; // NNBB we accept "resonances" even outside mass bounds
              maxprob = prob;
            }
            double deltaMass = fabs(mcomb-ResMass[ires]);
            if( deltaMass<minDeltaMass ){
              bestMassMuons = make_pair((*Muon1),(*Muon2));
              minDeltaMass = deltaMass;
            }
          }
        }
      }
    }
  }
  //If outside mass window (maxprob==0) then take the two muons with best invariant mass
  //(anyway they will not be used in the likelihood calculation, only to fill plots)
  if(!maxprob){
    if(bestMassMuons.first.charge()<0){
      recMuFromBestRes.first = bestMassMuons.first.p4();
      recMuFromBestRes.second = bestMassMuons.second.p4();
    }
    else{
      recMuFromBestRes.second = bestMassMuons.first.p4();
      recMuFromBestRes.first = bestMassMuons.second.p4();      
    }
  }

  return recMuFromBestRes;
}

pair <lorentzVector, lorentzVector> MuScleFitUtils::findGenMuFromRes( const Handle<HepMCProduct> & evtMC )
{
  const HepMC::GenEvent* Evt = evtMC->GetEvent();
  pair<lorentzVector,lorentzVector> muFromRes;
  //Loop on generated particles
  for (HepMC::GenEvent::particle_const_iterator part=Evt->particles_begin(); 
       part!=Evt->particles_end(); part++) {
    if (fabs((*part)->pdg_id())==13 && (*part)->status()==1) {
      bool fromRes = false;
      for (HepMC::GenVertex::particle_iterator mother = (*part)->production_vertex()->particles_begin(HepMC::ancestors);
	   mother != (*part)->production_vertex()->particles_end(HepMC::ancestors); ++mother) {
        unsigned int motherPdgId = (*mother)->pdg_id();

        // For sherpa the resonance is not saved. The muons from the resonance can be identified
        // by having as mother a muon of status 3.
        if( sherpa_ ) {
          if( motherPdgId == 13 && (*mother)->status() == 3 ) fromRes = true;
        }
        else {
          for( int ires = 0; ires < 6; ++ires ) {
            if( motherPdgId == motherPdgIdArray[ires] && resfind[ires] ) fromRes = true;
          }
        }
      }
      if(fromRes){
	if((*part)->pdg_id()==13)
	  muFromRes.first = (lorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
					   (*part)->momentum().pz(),(*part)->momentum().e()));
	else
	  muFromRes.second = (lorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
					    (*part)->momentum().pz(),(*part)->momentum().e()));
      }
    }
  }
  return muFromRes;
}

pair <lorentzVector, lorentzVector> MuScleFitUtils::findGenMuFromRes( const Handle<GenParticleCollection> & genParticles)
{
  pair<lorentzVector,lorentzVector> muFromRes;

  //Loop on generated particles
  for( GenParticleCollection::const_iterator part=genParticles->begin(); part!=genParticles->end(); ++part ) {
    if (fabs(part->pdgId())==13 && part->status()==1) {
      bool fromRes = false;
      unsigned int motherPdgId = part->mother()->pdgId();
      for( int ires = 0; ires < 6; ++ires ) {
	if( motherPdgId == motherPdgIdArray[ires] && resfind[ires] ) fromRes = true;
      }
      if(fromRes){
	if(part->pdgId()==13)
	  muFromRes.first = (lorentzVector(part->p4().px(),part->p4().py(),
					   part->p4().pz(),part->p4().e()));
	else
	  muFromRes.second = (lorentzVector(part->p4().px(),part->p4().py(),
					    part->p4().pz(),part->p4().e()));
      }
    }
  }
  return muFromRes;
}

pair <lorentzVector, lorentzVector> MuScleFitUtils::findSimMuFromRes( const Handle<HepMCProduct> & evtMC, const Handle<SimTrackContainer> & simTracks )
{
  //Loop on simulated tracks
  pair<lorentzVector, lorentzVector> simMuFromRes;
  for (SimTrackContainer::const_iterator simTrack=simTracks->begin(); simTrack!=simTracks->end(); ++simTrack) {
    //Chose muons
    if (fabs((*simTrack).type())==13) {
      //If tracks from IP than find mother
      if ((*simTrack).genpartIndex()>0) {
	HepMC::GenParticle* gp = evtMC->GetEvent()->barcode_to_particle ((*simTrack).genpartIndex()); 
        if( gp != 0 ) {

          for (HepMC::GenVertex::particle_iterator mother = gp->production_vertex()->particles_begin(HepMC::ancestors);
               mother!=gp->production_vertex()->particles_end(HepMC::ancestors); ++mother) {

            bool fromRes = false;
            unsigned int motherPdgId = (*mother)->pdg_id();
            for( int ires = 0; ires < 6; ++ires ) {
              if( motherPdgId == motherPdgIdArray[ires] && resfind[ires] ) fromRes = true;
            }
//  	  if( ((*mother)->pdg_id() == 23     && resfind[0]) ||
//               ((*mother)->pdg_id() == 443    && resfind[1]) ||
//               ((*mother)->pdg_id() == 100443 && resfind[2]) || 
//  	      ((*mother)->pdg_id() == 553    && resfind[3]) ||
//               ((*mother)->pdg_id() == 100553 && resfind[4]) ||
//               ((*mother)->pdg_id() == 200553 && resfind[5]) {
            if( fromRes ) {
              if(gp->pdg_id() == 13)
                simMuFromRes.first = lorentzVector(simTrack->momentum().px(),simTrack->momentum().py(),
                                                   simTrack->momentum().pz(),simTrack->momentum().e());
              else
                simMuFromRes.second = lorentzVector(simTrack->momentum().px(),simTrack->momentum().py(),
                                                    simTrack->momentum().pz(),simTrack->momentum().e()); 
            }
          }
        }
        else LogDebug("MuScleFitUtils") << "WARNING: no matching genParticle found for simTrack" << endl;
      }
    }
  }
  return simMuFromRes;
}

// Zero a few arrays
// -----------------
void MuScleFitUtils::cleanEstimator() {
  mzsum = 0.;
  isum = 0;
  for (int i=0; i<11; i++) {
    for (int j=0; j<100; j++) {
      f[i][j] = 0.;
      g[i][j] = 0.;
    }
  }
}

// Compute estimator h
// -------------------
void MuScleFitUtils::computeEstimator( const lorentzVector& recMu1, 
 				       const lorentzVector& recMu2,
                                       const double & Mass) {
  mzsum += Mass;
  isum += 1.;
  computeEstimator(recMu1,Mass);
  computeEstimator(recMu2,Mass);
  int ibin;

  // f[3], g[3]: DM versus muon DPhi
  // -------------------------------
  ibin = (int)((3.1415926-abs(abs(recMu1.phi()-recMu2.phi())-3.1415926))*100/3.1415926);
  if (ibin>=0 && ibin<100) {
    f[3][ibin] += Mass;
    g[3][ibin] += 1.;
  }
  // f[4], g[4]: DM versus muon DEta
  // -------------------------------
  ibin = (int)((6.+recMu1.eta()-recMu2.eta())*100/12.);
  if (ibin>=0 && ibin<100) {
    f[4][ibin] += Mass;
    g[4][ibin] += 1.;
  }
  // f[5], g[5]: DM versus negative muon Pt
  // --------------------------------------
  ibin = (int)recMu1.pt();
  if (ibin>=0 && ibin<100) {
    f[5][ibin] += Mass;
    g[5][ibin] += 1.;
  }
  // f[6], g[6]: DM versus positive muon Pt
  // --------------------------------------
  ibin = (int)recMu2.pt();
  if (ibin>=0 && ibin<100) {
    f[6][ibin] += Mass;
    g[6][ibin] += 1.;
  }
  // f[7], g[7]: DM versus negative muon Eta
  // ---------------------------------------
  ibin = 50+(int)(recMu1.eta()*100/6.);
  if (ibin>=0 && ibin<100) {
    f[7][ibin] += Mass;
    g[7][ibin] += 1.;
  }
  // f[8], g[8]: DM versus positive muon Eta
  // ---------------------------------------
  ibin = 50+(int)(recMu2.eta()*100/6.);
  if (ibin>=0 && ibin<100) {
    f[8][ibin] += Mass;
    g[8][ibin] += 1.;
  }
  // f[9], g[9]: DM versus negative muon Phi
  // ---------------------------------------
  ibin = (int)(recMu1.phi()*100/6.283);
  if (ibin>=0 && ibin<100) {
    f[9][ibin] += Mass;
    g[9][ibin] += 1.;
  }
  // f[10], g[10]: DM versus positive muon Phi
  // -----------------------------------------
  ibin = (int)(recMu2.phi()*100/6.283);
  if (ibin>=0 && ibin<100) {
    f[10][ibin] += Mass;
    g[10][ibin] += 1.;
  }
}

// Fill array elements depending on single muon quantities
// -------------------------------------------------------
void MuScleFitUtils::computeEstimator( const lorentzVector & recMu, const double & Mass) {
  // f[0], g[0]: DM versus muon Pt
  // ------------------------------
  int ibin = (int)recMu.pt();
  if (ibin>=0 && ibin<100) {
    f[0][ibin] += Mass;
    g[0][ibin] += 1.;
  }
  // f[1], g[1]: DM versus muon Eta
  // ------------------------------
  ibin = 50+(int)(recMu.eta()*100/6.);
  if (ibin>=0 && ibin<100) {
    f[1][ibin] += Mass;
    g[1][ibin] += 1.;
  }
  // f[2], g[2]: DM versus muon Phi
  // ------------------------------
  ibin = (int)(recMu.phi()*100/6.283);
  if (ibin>=0 && ibin<100) {
    f[2][ibin] += Mass;
    g[2][ibin] += 1.;
  }
}

// Final operations for h estimator
// --------------------------------
void MuScleFitUtils::returnEstimator() {

  TString Names[11] = {"Muon Pt", "Muon eta", "Muon Phi", "Muon DPhi", "Muon DEta", 
		       "Mu- Pt", "Mu+ Pt", "Mu- Eta", "Mu+ Eta", "Mu- Phi", "Mu+ Phi"};
  double h[11];
  double mzave = mzsum/isum;
  for (int i=0; i<11; i++) {
    h[i] = 0;
    for (int j=0; j<100; j++) {
      f[i][j] -= mzave*g[i][j];                          // Return to residuals around zero
      if (g[i][j]>0) h[i] += abs(f[i][j])/g[i][j]/100.;
    }
    cout << Names[i] << ": h function is " << h[i] << endl;
  }
}

// Resolution smearing function called to worsen muon Pt resolution at start
// -------------------------------------------------------------------------
lorentzVector MuScleFitUtils::applySmearing (const lorentzVector& muon)
{
  double pt = muon.Pt();
  double eta = muon.Eta();
  double phi = muon.Phi();
  double E = muon.E();

  double y[7];
  for (int i=0; i<SmearType+3; i++) {
    y[i] = x[i][goodmuon%10000];
  }

  // Use the smear function selected in the constructor
  smearFunction->smear( pt, eta, phi, y, parSmear );

  if (debug>9) {
    cout << "Smearing Pt,eta,phi = " << pt << " " <<  eta << " " 
	 << phi << "; x = ";
    for (int i=0; i<SmearType+3; i++) {
      cout << y[i];
    }
    cout << endl;
  }

  double ptEtaPhiE[4] = {pt, eta, phi, E};
  return( fromPtEtaPhiToPxPyPz(ptEtaPhiE) );
}

// Biasing function called to modify muon Pt scale at the start.
// -------------------------------------------------------------
lorentzVector MuScleFitUtils::applyBias( const lorentzVector& muon, const int chg )
{
  double ptEtaPhiE[4] = {muon.Pt(),muon.Eta(),muon.Phi(),muon.E()};

  if (MuScleFitUtils::debug>1) cout << "pt before bias = " << ptEtaPhiE[0] << endl;

  // Use functors (although not with the () operator)
  // Note that we always pass pt, eta and phi, but internally only the needed
  // values are used.
  // The functors used are takend from the same group used for the scaling
  // thus the name of the method used is "scale".
  ptEtaPhiE[0] = biasFunction->scale(ptEtaPhiE[0], ptEtaPhiE[1], ptEtaPhiE[2], chg, MuScleFitUtils::parBias);

  if (MuScleFitUtils::debug>1) cout << "pt after bias = " << ptEtaPhiE[0] << endl;

  return( fromPtEtaPhiToPxPyPz(ptEtaPhiE) );
}

// Version of applyScale accepting a vector<double> of parameters
// --------------------------------------------------------------
lorentzVector MuScleFitUtils::applyScale (const lorentzVector& muon,
                                          const vector<double> & parval, const int chg)
{
  double * p = new double[(int)(parval.size())];
  // Replaced by auto_ptr, which handles delete at the end
  // std::auto_ptr<double> p(new double[(int)(parval.size())]);
  // Removed auto_ptr, check massResolution for an explanation.
  int id = 0;
  for (vector<double>::const_iterator it=parval.begin(); it!=parval.end(); ++it, ++id) {
    //(&*p)[id] = *it; 
    // Also ok would be (p.get())[id] = *it;
    p[id] = *it; 
  }
  lorentzVector tempScaleVec( applyScale (muon, p, chg) );
  delete[] p;
  return tempScaleVec;
}

// This is called by the likelihood to "taste" different values for additional corrections
// ---------------------------------------------------------------------------------------
lorentzVector MuScleFitUtils::applyScale (const lorentzVector& muon,
                                          double* parval, const int chg)
{
  double ptEtaPhiE[4] = {muon.Pt(),muon.Eta(),muon.Phi(),muon.E()};
  int shift = parResol.size();

  if (MuScleFitUtils::debug>1) cout << "pt before scale = " << ptEtaPhiE[0] << endl;

  // the address of parval[shift] is passed as pointer to double. Internally it is used as a normal array, thus:
  // array[0] = parval[shift], array[1] = parval[shift+1], ...
  ptEtaPhiE[0] = scaleFunction->scale(ptEtaPhiE[0], ptEtaPhiE[1], ptEtaPhiE[2], chg, &(parval[shift]));

  if (MuScleFitUtils::debug>1) cout << "pt after scale = " << ptEtaPhiE[0] << endl;

  return( fromPtEtaPhiToPxPyPz(ptEtaPhiE) );
}


// Useful function to convert 4-vector coordinates
// -----------------------------------------------
lorentzVector MuScleFitUtils::fromPtEtaPhiToPxPyPz( const double* ptEtaPhiE )
{
  double px = ptEtaPhiE[0]*cos(ptEtaPhiE[2]);
  double py = ptEtaPhiE[0]*sin(ptEtaPhiE[2]);
  double tmp = 2*atan(exp(-ptEtaPhiE[1]));
  double pz = ptEtaPhiE[0]*cos(tmp)/sin(tmp);
  double E  = sqrt(px*px+py*py+pz*pz+muMass*muMass);

  // lorentzVector corrMu(px,py,pz,E);
  // To fix memory leaks, this is to be substituted with
  // std::auto_ptr<lorentzVector> corrMu(new lorentzVector(px, py, pz, E));

  return lorentzVector(px,py,pz,E);
}

// Dimuon mass
// -----------
double MuScleFitUtils::invDimuonMass( const lorentzVector& mu1, 
                                             const lorentzVector& mu2 )
{
  return (mu1+mu2).mass();
}

// Mass resolution - version accepting a vector<double> parval
// -----------------------------------------------------------
double MuScleFitUtils::massResolution( const lorentzVector& mu1,
                                       const lorentzVector& mu2,
                                       const vector<double> & parval )
{
  // double * p = new double[(int)(parval.size())];
  // Replaced by auto_ptr, which handles delete at the end
  // --------- //
  // ATTENTION //
  // --------- //
  // auto_ptr calls delete, not delete[] and thus it must
  // not be used with arrays. There are alternatives see
  // e.g.: http://www.gotw.ca/gotw/042.htm. The best
  // alternative seems to be to switch to vector though.
  // std::auto_ptr<double> p(new double[(int)(parval.size())]);
  
  double * p = new double[(int)(parval.size())];
  vector<double>::const_iterator it = parval.begin();
  int id = 0;
  for ( ; it!=parval.end(); ++it, ++id) {
    // (&*p)[id] = *it;
    p[id] = *it;
  }
  double massRes = massResolution (mu1, mu2, p);
  delete[] p;
  return massRes;
}

double MuScleFitUtils::massResolution( const lorentzVector& mu1,
                                       const lorentzVector& mu2,
                                       double* parval )
{
  // We use the following formula:
  // 
  // M = sqrt ( (E1+E2)^2 - (P1+P2)^2 ) 
  //   
  // where we express E and P as a function of Pt, phi, and theta:
  //
  // E  = sqrt ( Pt^2*(1+cotg(theta)^2) + M_mu^2 )
  // Px = Pt*cos(phi), Py = Pt*sin(phi), Pz = Pt*cotg(theta)
  // 
  // from which we find 
  //
  // M = sqrt( 2*M_mu^2 + 2*sqrt(Pt1^2/sin(theta1)^2 + M_mu^2)*sqrt(Pt2^2/sin(theta2)^2 + M_mu^2) -
  //           2*Pt1*Pt2* ( cos(phi1-phi2) + cotg(theta1)*cotg(theta2) ) )
  //
  // and derive WRT Pt1, Pt2, phi1, phi2, theta1, theta2 to get the resolution.
  // --------------------------------------------------------------------------
  double mass   = (mu1+mu2).mass();
  double pt1    = mu1.Pt();
  double phi1   = mu1.Phi();
  double eta1   = mu1.Eta();
  double theta1 = 2*atan(exp(-eta1));
  // double cotgTheta1 = cos(theta1)/sin(theta1);
  double pt2    = mu2.Pt();
  double phi2   = mu2.Phi();
  double eta2   = mu2.Eta();
  double theta2 = 2*atan(exp(-eta2));
  // double cotgTheta2 = cos(theta2)/sin(theta2);

  // double mass_check = sqrt(2*mMu2+2*sqrt(pow(pt1/sin(theta1),2)+mMu2)*sqrt(pow(pt2/sin(theta2),2)+mMu2)-
  //			   2*pt1*pt2*(cos(phi1-phi2)+1/(tan(theta1)*tan(theta2))));

  // ATTENTION: need to compute 1/tan(theta) as cos(theta)/sin(theta) because the latter diverges for theta=pi/2
  // -----------------------------------------------------------------------------------------------------------
  double dmdpt1  = (pt1/pow(sin(theta1),2)*sqrt((pow(pt2/sin(theta2),2)+mMu2)/(pow(pt1/sin(theta1),2)+mMu2))- 
		    pt2*(cos(phi1-phi2)+cos(theta1)*cos(theta2)/(sin(theta1)*sin(theta2))))/mass;
  double dmdpt2  = (pt2/pow(sin(theta2),2)*sqrt((pow(pt1/sin(theta1),2)+mMu2)/(pow(pt2/sin(theta2),2)+mMu2))- 
		    pt1*(cos(phi2-phi1)+cos(theta2)*cos(theta1)/(sin(theta2)*sin(theta1))))/mass;
  double dmdphi1 = pt1*pt2/mass*sin(phi1-phi2);
  double dmdphi2 = pt2*pt1/mass*sin(phi2-phi1);
  // double dmdtheta1 = (-pow(pt1/sin(theta1),2)/tan(theta1)*
  //	 	        sqrt((pow(pt2/sin(theta2),2)+mMu2)/(pow(pt1/sin(theta1),2)+mMu2))+
  //		        2*pt1*pt2/(tan(theta2)*pow(sin(theta1),2)))/mass;
  // double dmdtheta2 = (-pow(pt2/sin(theta2),2)/tan(theta2)*
  //		        sqrt((pow(pt1/sin(theta1),2)+mMu2)/(pow(pt2/sin(theta2),2)+mMu2))+
  //	                2*pt2*pt1/(tan(theta1)*pow(sin(theta2),2)))/mass;
  double dmdcotgth1 = (pt1*pt1*cos(theta1)/sin(theta1)*
                       sqrt((pow(pt2/sin(theta2),2)+mMu2)/(pow(pt1/sin(theta1),2)+mMu2)) - 
		       pt1*pt2*cos(theta2)/sin(theta2))/mass;
  double dmdcotgth2 = (pt2*pt2*cos(theta2)/sin(theta2)*
                       sqrt((pow(pt1/sin(theta1),2)+mMu2)/(pow(pt2/sin(theta2),2)+mMu2)) - 
		       pt2*pt1*cos(theta1)/sin(theta1))/mass;

  if( debugMassResol_ ) {
    massResolComponents.dmdpt1 = dmdpt1;
    massResolComponents.dmdpt2 = dmdpt2;
    massResolComponents.dmdphi1 = dmdphi1;
    massResolComponents.dmdphi2 = dmdphi2;
    massResolComponents.dmdcotgth1 = dmdcotgth1;
    massResolComponents.dmdcotgth2 = dmdcotgth2;
  }

  // Resolution parameters:
  // ----------------------
  double sigma_pt1 = resolutionFunction->sigmaPt( pt1,eta1,parval );
  double sigma_pt2 = resolutionFunction->sigmaPt( pt2,eta2,parval );
  double sigma_phi1 = resolutionFunction->sigmaPhi( pt1,eta1,parval );
  double sigma_phi2 = resolutionFunction->sigmaPhi( pt2,eta2,parval );
  double sigma_cotgth1 = resolutionFunction->sigmaCotgTh( pt1,eta1,parval );
  double sigma_cotgth2 = resolutionFunction->sigmaCotgTh( pt2,eta2,parval );

  // Sigma_Pt is defined as a relative sigmaPt/Pt for this reason we need to
  // multiply it by pt.
  double mass_res = sqrt(pow(dmdpt1*sigma_pt1*pt1,2)+pow(dmdpt2*sigma_pt2*pt2,2)+
  			 pow(dmdphi1*sigma_phi1,2)+pow(dmdphi2*sigma_phi2,2)+
  			 pow(dmdcotgth1*sigma_cotgth1,2)+pow(dmdcotgth2*sigma_cotgth2,2));

  // double mass_res = sqrt(pow(dmdpt1*sigma_pt1*pt1,2)+pow(dmdpt2*sigma_pt2*pt2,2));

  if (debug>19) { 
    cout << "  Pt1=" << pt1 << " phi1=" << phi1 << " cotgth1=" << cos(theta1)/sin(theta1) << " - Pt2=" << pt2 
	 << " phi2=" << phi2 << " cotgth2=" << cos(theta2)/sin(theta2) << endl; 
    cout << " P[0]=" 
	 << parval[0] << " P[1]=" << parval[1] << "P[2]=" << parval[2] << " P[3]=" << parval[3] << endl;
    cout << "  Dmdpt1= " << dmdpt1 << " dmdpt2= " << dmdpt2 << " sigma_pt1=" 
	 << sigma_pt1 << " sigma_pt2=" << sigma_pt2 << endl;
    cout << "  Dmdphi1= " << dmdphi1 << " dmdphi2= " << dmdphi2 << " sigma_phi1=" 
	 << sigma_phi1 << " sigma_phi2=" << sigma_phi2 << endl;
    cout << "  Dmdcotgth1= " << dmdcotgth1 << " dmdcotgth2= " << dmdcotgth2 
	 << " sigma_cotgth1=" 
	 << sigma_cotgth1 << " sigma_cotgth2=" << sigma_cotgth2 << endl;
    cout << "  Mass resolution (pval) for muons of Pt = " << pt1 << " " << pt2 
	 << " : " << mass << " +- " << mass_res << endl;  
  }

  // Debug cout
  // ----------
  bool didit = false;
  for (int ires=0; ires<6; ires++) {
    if (!didit && resfind[ires]>0 && fabs(mass-ResMass[ires])<ResHalfWidth[ires]) {
      if (mass_res>ResMaxSigma[ires] && counter_resprob<100) {
	counter_resprob++;
	LogDebug("MuScleFitUtils") << "RESOLUTION PROBLEM: ires=" << ires << endl;
	// cout << "RESOLUTION PROBLEM: ires=" << ires << endl;
// 	cout << "---------------------------" << endl;
// 	cout << "  Pt1=" << pt1 << " phi1=" << phi1 << " cotgth1=" << cos(theta1)/sin(theta1) << " - Pt2=" << pt2 
// 	     << " phi2=" << phi2 << " cotgth2=" << cos(theta2)/sin(theta2) << endl; 
// 	if (ResolFitType==1) 
// 	  cout << " P[0]=" 
// 	       << parval[0] << " P[1]=" << parval[1] << "P[2]=" << parval[2] << " P[3]=" << parval[3] 
// 	       << " P[4]=" << parval[4] << " P[5]=" << parval[5] << endl;
// 	if (ResolFitType==2)
// 	  cout << " P[0]=" 
// 	       << parval[0] << " P[1]=" << parval[1] << "P[2]=" << parval[2] << " P[3]=" << parval[3] << endl;
// 	cout << "  Dmdpt1= " << dmdpt1 << " dmdpt2= " << dmdpt2 << " sigma_pt1=" 
// 	     << sigma_pt1 << " sigma_pt2=" << sigma_pt2 << endl;
// 	cout << "  Dmdphi1= " << dmdphi1 << " dmdphi2= " << dmdphi2 << " sigma_phi1=" 
// 	     << sigma_phi1 << " sigma_phi2=" << sigma_phi2 << endl;
// 	cout << "  Dmdcotgth1= " << dmdcotgth1 << " dmdcotgth2= " << dmdcotgth2 
// 	     << " sigma_cotgth1=" 
// 	     << sigma_cotgth1 << " sigma_cotgth2=" << sigma_cotgth2 << endl;
// 	cout << "  Mass resolution (pval) for muons of Pt = " << pt1 << " " << pt2 
// 	     << " : " << mass << " +- " << mass_res << endl;  
// 	cout << "---------------------------" << endl;
	didit = true;
      }
    }
  }
  
  return mass_res;
}

// Mass probability - version with linear background included, accepts vector<double> parval
// -----------------------------------------------------------------------------------------
double MuScleFitUtils::massProb( const double & mass, const double & rapidity, const double & massResol, const vector<double> & parval )
{
#ifdef USE_CALLGRIND
  CALLGRIND_START_INSTRUMENTATION;
#endif

  double * p = new double[(int)(parval.size())];
  // Replaced by auto_ptr, which handles delete at the end
  // Removed auto_ptr, check massResolution for an explanation.
  // std::auto_ptr<double> p(new double[(int)(parval.size())]);

  vector<double>::const_iterator it = parval.begin();
  int id = 0;
  for ( ; it!=parval.end(); ++it, ++id) {
    // (&*p)[id] = *it;
    p[id] = *it;
  }
  // p must be passed by value as below:
  double massProbability = massProb (mass, rapidity, massResol, p);
  delete[] p;

#ifdef USE_CALLGRIND
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif

  return massProbability;
}

/**
 * After the introduction of the rapidity bins for the Z the probability method works in the following way:
 * - if passing iRes == 0, iY is used to select the rapidity bin
 * - if passing iRes != 0, iY is used to select the resonance
 */
double MuScleFitUtils::probability( const double & mass, const double & massResol,
                                    const double GLvalue[][1001][1001], const double GLnorm[][1001],
                                    const int iRes, const int iY )
{
  if( iRes == 0 && iY > 23 ) {
    cout << "WARNING: rapidity bin selected = " << iY << " but there are only histograms for the first 24 bins" << endl;
  }

  double PS = 0.;
  bool insideProbMassWindow = true;
  // Interpolate the four values of GLZValue[] in the 
  // grid square within which the (mass,sigma) values lay 
  // ----------------------------------------------------
  // This must be done with respect to the width used in the computation of the probability distribution,
  // so that the bin 0 really matches the bin 0 of that distribution.
  double fracMass = (mass-(ResMass[iRes]-ResHalfWidth[iRes]))/(2*ResHalfWidth[iRes]);
  if (debug>1) cout << setprecision(9)<<"mass ResMass[iRes] ResHalfWidth[iRes] ResHalfWidth[iRes]"
                    << mass << " "<<ResMass[iRes]<<" "<<ResHalfWidth[iRes]<<" "<<ResHalfWidth[iRes]<<endl;
  int iMassLeft  = (int)(fracMass*(double)nbins);
  int iMassRight = iMassLeft+1;
  double fracMassStep = (double)nbins*(fracMass - (double)iMassLeft/(double)nbins);
  if (debug>1) cout<<"nbins iMassLeft fracMass "<<nbins<<" "<<iMassLeft<<" "<<fracMass<<endl;

  // Simple protections for the time being: the region where we fit should not include
  // values outside the boundaries set by ResMass-ResHalfWidth : ResMass+ResHalfWidth
  // ---------------------------------------------------------------------------------
  if (iMassLeft<0) {
    LogInfo("probability") << "WARNING: fracMass=" << fracMass << ", iMassLeft=" 
                           << iMassLeft << "; mass = " << mass << " and bounds are " << ResMass[iRes]-ResHalfWidth[iRes] 
                           << ":" << ResMass[iRes]+ResHalfWidth[iRes] << " - iMassLeft set to 0" << endl;
    iMassLeft  = 0;
    iMassRight = 1;
    insideProbMassWindow = false;
  }
  if (iMassRight>nbins) {
    LogInfo("probability") << "WARNING: fracMass=" << fracMass << ", iMassRight=" 
                           << iMassRight << "; mass = " << mass << " and bounds are " << ResMass[iRes]-ResHalfWidth[iRes] 
                           << ":" << ResMass[iRes]+ResHalfWidth[iRes] << " - iMassRight set to " << nbins-1 << endl;
    iMassLeft  = nbins-1;
    iMassRight = nbins;
    insideProbMassWindow = false;
  }
  double fracSigma = (massResol/ResMaxSigma[iRes]);
  int iSigmaLeft = (int)(fracSigma*(double)nbins);
  int iSigmaRight = iSigmaLeft+1;
  double fracSigmaStep = (double)nbins * (fracSigma - (double)iSigmaLeft/(double)nbins);
  // cout << "massResol = " << massResol << endl;
  // cout << "ResMaxSigma["<<iRes<<"] = " << ResMaxSigma[iRes] << endl;
  // cout << "fracSigma = " << fracSigma << endl;
  // cout << "nbins = " << nbins << endl;
  // cout << "ISIGMALEFT = " << iSigmaLeft << endl;
  // cout << "ISIGMARIGHT = " << iSigmaRight << endl;
  // cout << "fracSigmaStep = " << fracSigmaStep << endl;

  // Simple protections for the time being: they should not affect convergence, since
  // ResMaxSigma is set to very large values, and if massResol exceeds them the fit 
  // should not get any prize for that (for large sigma, the prob. distr. becomes flat)
  // ----------------------------------------------------------------------------------
  if (iSigmaLeft<0) { 
    LogInfo("probability") << "WARNING: fracSigma = " << fracSigma << ", iSigmaLeft=" 
                           << iSigmaLeft << ", with massResol = " << massResol << " and ResMaxSigma[iRes] = "
                           << ResMaxSigma[iRes] << " -  iSigmaLeft set to 0" << endl;
    iSigmaLeft  = 0;
    iSigmaRight = 1;
  }
  if (iSigmaRight>nbins ) { 
    if (counter_resprob<100)
      LogInfo("probability") << "WARNING: fracSigma = " << fracSigma << ", iSigmaRight=" 
                             << iSigmaRight << ", with massResol = " << massResol << " and ResMaxSigma[iRes] = "
                             << ResMaxSigma[iRes] << " -  iSigmaRight set to " << nbins-1 << endl;
    iSigmaLeft  = nbins-1;
    iSigmaRight = nbins;
  }

  // If f11,f12,f21,f22 are the values at the four corners, one finds by linear interpolation the
  // formula below for PS
  // --------------------------------------------------------------------------------------------
  if( insideProbMassWindow ) {
    double f11 = 0.;
    if (GLnorm[iY][iSigmaLeft]>0)
      f11 = GLvalue[iY][iMassLeft][iSigmaLeft] / GLnorm[iY][iSigmaLeft];
    double f12 = 0.;
    if (GLnorm[iY][iSigmaRight]>0)
      f12 = GLvalue[iY][iMassLeft][iSigmaRight] / GLnorm[iY][iSigmaRight];
    double f21 = 0.;
    if (GLnorm[iY][iSigmaLeft]>0)
      f21 = GLvalue[iY][iMassRight][iSigmaLeft] / GLnorm[iY][iSigmaLeft];
    double f22 = 0.;
    if (GLnorm[iY][iSigmaRight]>0)
      f22 = GLvalue[iY][iMassRight][iSigmaRight] / GLnorm[iY][iSigmaRight];
    PS = f11 + (f12-f11)*fracSigmaStep + (f21-f11)*fracMassStep +
      (f22-f21-f12+f11)*fracMassStep*fracSigmaStep;
    if (PS>0.1 || debug>1) LogDebug("MuScleFitUtils") << "iRes = 0 " << " PS=" << PS << " f11,f12,f21,f22="
                                                      << f11 << " " << f12 << " " << f21 << " " << f22 << " "
                                                      << " fSS=" << fracSigmaStep << " fMS=" << fracMassStep << " iSL, iSR="
                                                      << iSigmaLeft << " " << iSigmaRight << " GLV,GLN="
                                                      << GLvalue[iY][iMassLeft][iSigmaLeft]
                                                      << " " << GLnorm[iY][iSigmaLeft] << endl;
  }
  else {
    LogInfo("probability") << "outside mass probability window. Setting PS["<<iRes<<"] = 0" << endl;
  }
  return PS;
}

// Mass probability - version with linear background included
// ----------------------------------------------------------
double MuScleFitUtils::massProb( const double & mass, const double & rapidity, const double & massResol, double * parval ) {

  // double MuScleFitUtils::massProb (double mass, double massResol,  std::auto_ptr<double> parval) {

  // This routine computes the likelihood that a given measured mass "measMass" is
  // the result of a reference mass ResMass[] if the resolution
  // expected for the two muons is massResol. 
  // This version includes two parameters (the last two in parval, by default)
  // to size up the background fraction and its relative normalization with respect
  // to the signal shape.
  //
  // We model the signal probability with a Lorentz L(M,H) of resonance mass M and natural width H
  // convoluted with a gaussian G(m,s) of measured mass m and expected mass resolution s,
  // by integrating over the intersection of the supports of L and G (which can be made to coincide with
  // the region where L is non-zero, given that H<<s in most cases) the product L(M,H)*G(m,s) dx as follows:
  //
  //   GL(m,s) = Int(M-10H,M+10H) [ L(x-M,H) * G(x-m,s) ] dx
  //
  // The above convolution is computed numerically by an independent root macro, Probs.C, which outputs
  // the values in six 1001x1001 grids, one per resonance. 
  //
  // NB THe following block of explanations for background models is outdated, see detailed
  // explanations where the code computes PB.
  // +++++++++++++++++++++++
  //   For the background, instead, we have two choices: a linear and an exponential model.
  //     * For the linear model, we choose a one-parameter form whereby the line is automatically normalized
  //       in the support [x1,x2] where we defined our "signal region", as follows:
  //     
  //         B(x;b) = 1/(x2-x1) + {x - (x2+x1)/2} * b
  //
  //       Defined as above, B(x) is a line passing for the point of coordinates (x1+x2)/2, 1/(x2-x1),
  //       whose slope b has as support the interval ( -2/[(x1-x2)*(x1+x2)], 2/[(x1-x2)*(x1+x2)] )
  //       so that B(x) is always positive-definite in [x1,x2].
  //
  //     * For the exponential model, we define B(x;b) as
  //  
  //         B(x;b) = b * { exp(-b*x) / [exp(-b*x1)-exp(-b*x2)] }
  // 
  //       This one-parameter definition is automatically normalized to unity in [x1,x2], with a parameter
  //       b which has to be positive in order for the slope to be negative. 
  //       Please note that this model is not useful in most circumstances; a more useful form would be one
  //       which included a linear component.
  // ++++++++++++++++++++++
  //
  // Once GL(m,s) and B(x;b) are defined, we introduce a further parameter a, such that we can have the
  // likelihood control the relative fraction of signal and background. We first normalize GL(m,s) for
  // any given s by taking the integral
  //
  //   Int(x1,x2) GL(m,s) dm = K_s
  //
  // We then define the probability as 
  //  
  //   P(m,s,a,b) = GL(m,s)/K_s * a  +  B(x,b) * (1-a)
  //
  // with a taking on values in the interval [0,1].
  // Defined as above, the probability is well-behaved, in the sense that it has a value between 0 and 1,
  // and the four parameters m,s,a,b fully control its shape.
  // 
  // It is to be noted that the formulation above requires the computation of two rather time-consuming
  // integrals. The one defining GL(m,s) can be stored in a TH2D and loaded by the constructor from a
  // file suitably prepared, and this will save loads of computing time.
  // ----------------------------------------------------------------------------------------------------

  double P = 0.;
  int shift = parResol.size() + parScale.size();
  double Bgrp1 = 0.;
  //   double Bgrp2 = 0.; 
  //   double Bgrp3 = 0.;
  
  // NB defined as below, P is a non-rigorous "probability" that we observe
  // a dimuon mass "mass", given ResMass[], gamma, and massResol. It is what we need for the
  // fit which finds the best resolution parameters, though. A definition which is
  // more properly a probability is given below (in massProb2()), where the result
  // cannot be used to fit resolution parameters because the fit would always prefer
  // to set the res parameters to the minimum possible value (best resolution), 
  // to have a probability close to one of observing any mass.
  // -------------------------------------------------------------------------------

  // Determine what resonance(s) we have to deal with
  // NB for now we assume equal xs for each resonance
  // so we do not assign them different weights
  // ------------------------------------------------
  double PS[6] = {0.};
  bool resConsidered[6] = {false};
  int nres = 0;  // number of resonances contributing here

  // Used to skip resonances when computing the background
  // bool noBackground = false;

  // Factor to take into account the superposition of different resonances mass windows. In that case
  // the background should also be weighted multiple times, as the probabilities of the different resonances are all
  // normalized independently and their sum is not.
  int superpositionFactor = 0;

  // First check the Z, which is divided in 40 rapidity bins
  // NB max value of Z rapidity to be considered is 4. here
  // -------------------------------------------------------

  // ATTENTION: it should be:
  // ------------------------
  // First check the Z, which is divided in 24 rapidity bins
  // NB max value of Z rapidity to be considered is 2.4 here
  // -------------------------------------------------------

  // Do this only if we want to use the rapidity bins for the Z
  if( MuScleFitUtils::rapidityBinsForZ_ ) {
    // ATTENTION: cut on Z rapidity at 2.4 since we only have histograms up to that value
    pair<double, double> windowFactors = backgroundHandler->windowFactors( doBackgroundFit[loopCounter], 0 );
    if( resfind[0]>0 && checkMassWindow( mass, 0,
                                         backgroundHandler->resMass( doBackgroundFit[loopCounter], 0 ),
                                         windowFactors.first, windowFactors.second ) && fabs(rapidity)<2.4 ) {
      int iY = (int)(fabs(rapidity)*10.);
      resConsidered[0] = true;
      nres += 1;

      if (MuScleFitUtils::debug>1) cout << "massProb:resFound = 0, rapidity bin =" << iY << endl;

      // In this case the last value is the rapidity bin
      PS[0] = probability(mass, massResol, GLZValue, GLZNorm, 0, iY);

      // We are inside the current resonance mass window, check if we are also inside any other resonance mass window.
      for( int otherRes = 0; otherRes < 6; ++otherRes ) {
        if( otherRes != 0 ) {
          if( checkMassWindow( mass, 0,
                               backgroundHandler->resMass( doBackgroundFit[loopCounter], 0 ),
                               windowFactors.first, windowFactors.second ) ) ++superpositionFactor;
        }
      }
    }
  }
  // Next check the other resonances
  // -------------------------------
  int firstRes = 1;
  if( !MuScleFitUtils::rapidityBinsForZ_ ) firstRes = 0;
  for( int ires=firstRes; ires<6; ++ires ) {

    // cout << "ires = " << ires << endl;

    // Changed to test the background. This way the outside of the region is also used to determine the
    // parameters of the background function.
    if( resfind[ires] > 0 ) {

      // First is left, second is right (returns (1,1) in the case of resonances, it could be improved avoiding the call in this case)
      pair<double, double> windowFactor = backgroundHandler->windowFactors( doBackgroundFit[loopCounter], ires );
      if( checkMassWindow(mass, ires, backgroundHandler->resMass( doBackgroundFit[loopCounter], ires ),
                          windowFactor.first, windowFactor.second) ) {

        resConsidered[ires] = true;
        nres += 1;

        if (MuScleFitUtils::debug>1) cout << "massProb:resFound = " << ires << endl;

        // In this case the rapidity value is instead the resonance index again.
        PS[ires] = probability(mass, massResol, GLValue, GLNorm, ires, ires);

        // We are inside the current resonance mass window, check if we are also inside any other resonance mass window.
        for( int otherRes = 0; otherRes < 6; ++otherRes ) {
          if( otherRes != ires ) {
            if( checkMassWindow( mass, otherRes, backgroundHandler->resMass( doBackgroundFit[loopCounter], ires ),
                                 windowFactor.first, windowFactor.second ) ) ++superpositionFactor;
          }
        }
      }
    }
  }

  double PB = 0.;

  double PStot = 0.;
  for (int ires=0; ires<6; ++ires) {
    if (resfind[ires]>0) {

      // The first parameter is for the handler. The rest are forwarded to the specific backgroundFunction.
      // The parval is shifted so that the correct parameters are passed (depends on the function number and if we are fitting or not the background).
      pair<double, double> bgrResult = backgroundHandler->backgroundFunction( doBackgroundFit[loopCounter],
                                                                              &(parval[shift]), MuScleFitUtils::totalResNum, ires,
                                                                              resConsidered, ResMass, ResHalfWidth, MuonType, mass, nbins );
      if( debug>0 ) {
        cout << "doBackgroundFit["<<loopCounter<<"] = " << doBackgroundFit[loopCounter] << ", Bgrp1 = " << bgrResult.first << ", backgroundFunctionResult = " << bgrResult.second << endl;
      }
      Bgrp1 = bgrResult.first;
      PStot += PS[ires]*(1-Bgrp1);
      PB += Bgrp1*bgrResult.second;

    }
  }

  // P = PStot*Bgrp1 + PB*(1+superpositionFactor)*(1-Bgrp1);
  // P = PStot*Bgrp1 + PB*(1-Bgrp1);

  // Note that they alreay contain the fraction coefficients.
  P = PStot + PB;

  if (debug>0) cout << "mass = " << mass << ", P = " << P << ", Pstot = " << PStot << ", Pb = " << PB << ", bgrp1 = " << Bgrp1 << endl;

  if( MuScleFitUtils::signalProb_ != 0 && MuScleFitUtils::backgroundProb_ != 0 ) {
    MuScleFitUtils::signalProb_->SetBinContent(MuScleFitUtils::minuitLoop_, MuScleFitUtils::signalProb_->GetBinContent(MuScleFitUtils::minuitLoop_) + PStot);
    MuScleFitUtils::backgroundProb_->SetBinContent(MuScleFitUtils::minuitLoop_, MuScleFitUtils::backgroundProb_->GetBinContent(MuScleFitUtils::minuitLoop_) + PB);
  }
  return P;
}

// Method to check if the mass value is within the mass window of the i-th resonance.
bool MuScleFitUtils::checkMassWindow( const double & mass, const int ires, const double & resMass, const double & leftFactor, const double & rightFactor )
{
//   return( mass-ResMass[ires] > -leftFactor*massWindowHalfWidth[ires][MuonTypeForCheckMassWindow]
//           && mass-ResMass[ires] < rightFactor*massWindowHalfWidth[ires][MuonTypeForCheckMassWindow] );
  return( mass-resMass > -leftFactor*massWindowHalfWidth[ires][MuonTypeForCheckMassWindow]
          && mass-resMass < rightFactor*massWindowHalfWidth[ires][MuonTypeForCheckMassWindow] );
}

// Function that returns the weight for a muon pair
// ------------------------------------------------
double MuScleFitUtils::computeWeight( const double & mass, const int iev )
{
  // Compute weight for this event
  // -----------------------------
  double weight = 0.;
  
  // Take the highest-mass resonance within bounds
  // NB this must be revised once credible estimates of the relative xs of Y(1S), (2S), and (3S)
  // are made. Those are priors in the decision of which resonance to assign to an in-between event.
  // -----------------------------------------------------------------------------------------------
  for (int ires=0; ires<6; ires++) {
    if (resfind[ires]>0 && weight==0.) {
      pair<double, double> windowFactor = backgroundHandler->windowFactors( doBackgroundFit[loopCounter], ires );
      if( checkMassWindow(mass, ires, backgroundHandler->resMass( doBackgroundFit[loopCounter], ires ),
                          windowFactor.first, windowFactor.second) ) {
        weight = 1.0;
      }
    }
  }

  return weight;
}

// Likelihood minimization routine
// -------------------------------
void MuScleFitUtils::minimizeLikelihood()
{
  // Output file with fit parameters resulting from minimization
  // -----------------------------------------------------------
  ofstream FitParametersFile;
  FitParametersFile.open ("FitParameters.txt", ios::app);
  FitParametersFile << "Fitting with resolution, scale, bgr function # " 
		    << ResolFitType << " " << ScaleFitType << " " << BgrFitType 
		    << " - Iteration " << loopCounter << endl;

  // Fill parvalue and other vectors needed for the fitting
  // ------------------------------------------------------
  int parnumber = (int)(parResol.size()+parScale.size()+parBgr.size());

  // parvalue is a vector<vector<double> > storing all the parameters from all the loops
  parvalue.push_back(parResol);
  vector<double> *tmpVec = &(parvalue.back());

  // If this is not the first loop we want to start from neutral values
  // Otherwise the scale will start with values correcting again a bias
  // that is already corrected.
  if( scaleFitNotDone_ ) {
    tmpVec->insert( tmpVec->end(), parScale.begin(), parScale.end() );
    cout << "scaleFitNotDone: tmpVec->size() = " << tmpVec->size() << endl;
  }
  else {
    scaleFunction->resetParameters(tmpVec);
    cout << "scaleFitDone: tmpVec->size() = " << tmpVec->size() << endl;
  }
  tmpVec->insert( tmpVec->end(), parBgr.begin(), parBgr.end() );
  int i = 0;
  vector<double>::const_iterator it = tmpVec->begin();
  for( ; it != tmpVec->end(); ++it, ++i ) {
    cout << "tmpVec["<<i<<"] = " << *it << endl;
  }

  vector<int> parfix(parResolFix);
  parfix.insert( parfix.end(), parScaleFix.begin(), parScaleFix.end() );
  parfix.insert( parfix.end(), parBgrFix.begin(), parBgrFix.end() );

  vector<int> parorder(parResolOrder);
  parorder.insert( parorder.end(), parScaleOrder.begin(), parScaleOrder.end() );
  parorder.insert( parorder.end(), parBgrOrder.begin(), parBgrOrder.end() );

  // This is filled later
  vector<double> parerr(3*parnumber,0.);

  if (debug>19) {
    cout << "[MuScleFitUtils-minimizeLikelihood]: Parameters before likelihood " << endl;
    for (unsigned int i=0; i<(unsigned int)parnumber; i++) {
      cout << "  Par # " << i << " = " << parvalue[loopCounter][i] << " : free = " << parfix[i] << "; order = " 
	   << parorder[i] << endl;
    }
  }

  // Background rescaling from regions to resonances
  // -----------------------------------------------
  // If we are in a loop > 0 and we are not fitting the background, but we have fitted it in the previous iteration
  if( loopCounter > 0 && !(doBackgroundFit[loopCounter]) && doBackgroundFit[loopCounter-1] ) {
    // This rescales from regions to resonances
    int localMuonType = MuonType;
    if( MuonType > 2 ) localMuonType = 2;
    backgroundHandler->rescale( parBgr, ResMass, massWindowHalfWidth, localMuonType,
                                MuScleFitUtils::SavedPair);
  }

  // Init Minuit
  // -----------
  TMinuit rmin (parnumber);
  rminPtr_ = &rmin;
  rmin.SetFCN (likelihood);     // Unbinned likelihood
  // Standard initialization of minuit parameters:
  // sets input to be $stdin, output to be $stdout
  // and saving to a file.
  rmin.mninit (5, 6, 7);
  int ierror = 0;
  int istat;
  double arglis[4];
  arglis[0] = FitStrategy;      // Strategy 1 or 2
  // 1 standard
  // 2 try to improve minimum (slower)
  rmin.mnexcm ("SET STR", arglis, 1, ierror);

  // Set fit parameters
  // ------------------
  double * Start = new double[parnumber];
  double * Step  = new double[parnumber];
  double * Mini  = new double[parnumber];
  double * Maxi  = new double[parnumber];
  int * ind = new int[parnumber]; // Order of release of parameters 
  TString * parname = new TString[parnumber];

  MuScleFitUtils::resolutionFunctionForVec->setParameters( Start, Step, Mini, Maxi, ind, parname, parResol, parResolOrder, MuonType );

  // Take the number of parameters in the resolutionFunction and displace the arrays passed to the scaleFunction
  int resParNum = MuScleFitUtils::resolutionFunctionForVec->parNum();

  MuScleFitUtils::scaleFunctionForVec->setParameters( &(Start[resParNum]), &(Step[resParNum]), &(Mini[resParNum]), &(Maxi[resParNum]),
                                                      &(ind[resParNum]), &(parname[resParNum]), parScale, parScaleOrder, MuonType );

  int shift = resParNum + MuScleFitUtils::scaleFunctionForVec->parNum();

  MuScleFitUtils::backgroundHandler->setParameters( &(Start[shift]), &(Step[shift]), &(Mini[shift]), &(Maxi[shift]),
                                                    &(ind[shift]), &(parname[shift]), parBgr, parBgrOrder, MuonType );

  for( int ipar=0; ipar<parnumber; ++ipar ) {
    // cout << "parname["<<ipar<<"] = " << parname[ipar] << endl;
    // cout << "Start["<<ipar<<"] = " << Start[ipar] << endl;
    // cout << "Step["<<ipar<<"] = " << Step[ipar] << endl;
    // cout << "Mini["<<ipar<<"] = " << Mini[ipar] << endl;
    // cout << "Maxi["<<ipar<<"] = " << Maxi[ipar] << endl;
    rmin.mnparm( ipar, parname[ipar], Start[ipar], Step[ipar], Mini[ipar], Maxi[ipar], ierror );
  }

  // Do minimization
  // ---------------
  if (debug>19)
    cout << "[MuScleFitUtils-minimizeLikelihood]: Starting minimization" << endl;
  double fmin;
  double fdem;
  double errdef;
  int npari;
  int nparx;
  rmin.mnexcm ("call fcn", arglis, 1, ierror);

  // First, fix all parameters
  // -------------------------
  if (debug>19)
    cout << "[MuScleFitUtils-minimizeLikelihood]: First fix all parameters ...";
  for (int ipar=0; ipar<parnumber; ipar++) {
    rmin.FixParameter (ipar);
  }
  
  // Then release them in the specified order and refit
  // --------------------------------------------------
  if (debug>19) cout << " Then release them in order..." << endl;

  TString name;
  double pval;
  double pmin;
  double pmax;
  double errp;
  double errl;
  double errh;
  int ivar;
  double erro;
  double cglo;
  int n_times = 0;
  // n_times = number of loops required to unlock all parameters.

  int scaleParNum = scaleFunction->parNum();
  LogInfo("minimizeLikelihood") << "number of parameters for scaleFunction = " << scaleParNum << endl;
  LogInfo("minimizeLikelihood") << "number of parameters for resolutionFunction = " << resParNum << endl;
  LogInfo("minimizeLikelihood") << "number of parameters for resolutionFunction = " << parBgr.size() << endl;
  // LogInfo("minimizeLikelihood") << "number of parameters for backgroundFunction = " << backgroundFunction->parNum() << endl;

  for (int i=0; i<parnumber; i++) {
    // NB ind[] has been set as parorder[] previously
    if (n_times<ind[i]) {
      LogInfo("minimizeLikelihood") << "n_times = " << n_times << ", ind["<<i<<"] = " << ind[i] << ", scaleParNum = " << scaleParNum << ", doScaleFit["<<loopCounter<<"] = " << doScaleFit[loopCounter] << endl;
      // Set the n_times only if we will do the fit
      if ( i<resParNum ) {
        if( doResolFit[loopCounter] ) n_times = ind[i];
      }
      else if( i<resParNum+scaleParNum ) {
        if( doScaleFit[loopCounter] ) n_times = ind[i];
      }
      else if( doBackgroundFit[loopCounter] ) n_times = ind[i];
    }
  }
  for (int iorder=0; iorder<n_times+1; iorder++) { // Repeat fit n_times times
    bool somethingtodo = false;

    // Use parameters from cfg to select which fit to do
    // -------------------------------------------------
    if( doResolFit[loopCounter] ) {
      // Release resolution parameters and fit them
      // ------------------------------------------
      for( unsigned int ipar=0; ipar<parResol.size(); ++ipar ) {
	if( parfix[ipar]==0 && ind[ipar]==iorder ) {
	  rmin.Release( ipar );
	  somethingtodo = true;
	}
      }
    }
    if( doScaleFit[loopCounter] ) {
      // Release scale parameters and fit them
      // -------------------------------------
      for( unsigned int ipar=parResol.size(); ipar<parResol.size()+parScale.size(); ++ipar ) {      
	if( parfix[ipar]==0 && ind[ipar]==iorder ) { // parfix=0 means parameter is free
	  rmin.Release( ipar );
	  somethingtodo = true;
	}
      }
      scaleFitNotDone_ = false;
    }
    if( doBackgroundFit[loopCounter] ) {
      // Release background parameters and fit them
      // ------------------------------------------
      // for( int ipar=parResol.size()+parScale.size(); ipar<parnumber; ++ipar ) {
      // Free only the parameters for the regions, as the resonances intervals are never used to fit the background
      unsigned int parShift = parResol.size()+parScale.size();
      for( unsigned int ipar = parShift; ipar < parShift+backgroundHandler->regionsParNum(); ++ipar ) {
        // Release only those parameters for the resonances we are fitting
	if( parfix[ipar]==0 && ind[ipar]==iorder && backgroundHandler->unlockParameter(resfind, ipar - parShift) ) {
	  rmin.Release( ipar );
	  somethingtodo = true;
	}
      }
      // We are doing a background fit, count the number of events in the background regions
      double weight = 1.;
      backgroundHandler->countEventsInBackgroundWindows(SavedPair, weight);
    }

    // OK, now do minimization if some parameter has been released
    // -----------------------------------------------------------
    if( somethingtodo ) {
// #ifdef DEBUG
      minuitLoop_ = 0;
      char name[50];
      sprintf(name, "likelihoodInLoop_%d_%d", loopCounter, iorder);
      TH1D * tempLikelihoodInLoop = new TH1D(name, "likelihood value in minuit loop", 10000, 0, 10000);
      likelihoodInLoop_ = tempLikelihoodInLoop;
      char signalProbName[50];
      sprintf(signalProbName, "signalProb_%d_%d", loopCounter, iorder);
      TH1D * tempSignalProb = new TH1D(signalProbName, "signal probability", 10000, 0, 10000);
      signalProb_ = tempSignalProb;
      char backgroundProbName[50];
      sprintf(backgroundProbName, "backgroundProb_%d_%d", loopCounter, iorder);
      TH1D * tempBackgroundProb = new TH1D(backgroundProbName, "background probability", 10000, 0, 10000);
      backgroundProb_ = tempBackgroundProb;
// #endif
      rmin.mnexcm ("mini", arglis, 0, ierror);
// #ifdef DEBUG
      likelihoodInLoop_->Write();
      signalProb_->Write();
      backgroundProb_->Write();
      delete tempLikelihoodInLoop;
      delete tempSignalProb;
      delete tempBackgroundProb;
      likelihoodInLoop_ = 0;
      signalProb_ = 0;
      backgroundProb_ = 0;
// #endif
    }
    for (int ipar=0; ipar<parnumber; ipar++) {
      rmin.mnpout (ipar, name, pval, erro, pmin, pmax, ivar);
      rmin.mnerrs (ipar, errh, errl, errp, cglo);
      
      // Set error on params
      // -------------------
      if (errp!=0) {
	parerr[3*ipar] = errp;
      } else { 
	parerr[3*ipar] = (((errh)>(fabs(errl)))?(errh):(fabs(errl)));
	parerr[3*ipar+1] = errl;
	parerr[3*ipar+2] = errh;
      }
      if( ipar == 0 ) {
        FitParametersFile << " Resolution fit parameters:" << endl;
      }
      if( ipar == int(parResol.size()) ) {
        FitParametersFile << " Scale fit parameters:" << endl;
      }
      if( ipar == int(parResol.size()+parScale.size()) ) {
        FitParametersFile << " Background fit parameters:" << endl;
      }
      FitParametersFile << "  Results of the fit: parameter " << ipar << " has value " 
			<< pval << "+-" << parerr[3*ipar] << endl;
    }
    rmin.mnstat (fmin, fdem, errdef, npari, nparx, istat); // NNBB Commented for a check!
    FitParametersFile << endl;
  } // end loop on iorder
  FitParametersFile.close();

  // Save parameters in parvalue[] vector
  // ------------------------------------
  if (ierror!=0 && debug>0) 
    cout << "[MuScleFitUtils-minimizeLikelihood]: ierror!=0, bogus pars" << endl;
  for (int ipar=0; ipar<parnumber; ipar++) {
    rmin.mnpout (ipar, name, pval, erro, pmin, pmax, ivar);
    parvalue[loopCounter][ipar] = pval;
  }

  cout << "[MuScleFitUtils-minimizeLikelihood]: Parameters after likelihood " << endl;
  for (unsigned int ipar=0; ipar<(unsigned int)parnumber; ipar++) {
    cout << ipar << " " << parvalue[loopCounter][ipar] << " : free = " 
	 << parfix[ipar] << "; order = " << parorder[ipar] << endl;
  }

  // Put back parvalue into parResol, parScale, parBgr
  // -------------------------------------------------
  for (int i=0; i<(int)(parResol.size()); i++) {
    parResol[i] = parvalue[loopCounter][i];
  } 
  for (int i=0; i<(int)(parScale.size()); i++) {
    parScale[i] = parvalue[loopCounter][i+parResol.size()];
  } 
  for (int i=0; i<(int)(parBgr.size()); i++) {
    parBgr[i] = parvalue[loopCounter][i+parResol.size()+parScale.size()];
  }

  // Delete the arrays used to set some parameters
  delete[] Start;
  delete[] Step;
  delete[] Mini;
  delete[] Maxi;
  delete[] ind;
  delete[] parname;
}

// Likelihood function
// -------------------
extern "C" void likelihood( int& npar, double* grad, double& fval, double* xval, int flag ) {

  if (MuScleFitUtils::debug>19) cout << "[MuScleFitUtils-likelihood]: In likelihood function" << endl;

   const lorentzVector * recMu1;
   const lorentzVector * recMu2;
   lorentzVector corrMu1;
   lorentzVector corrMu2;

  if (MuScleFitUtils::debug>19) {
    int parnumber = (int)(MuScleFitUtils::parResol.size()+MuScleFitUtils::parScale.size()+
			  MuScleFitUtils::parBgr.size());
    cout << "[MuScleFitUtils-likelihood]: Looping on tree with ";
    for (int ipar=0; ipar<parnumber; ipar++) {
      cout << "Parameter #" << ipar << " with value " << xval[ipar] << " ";
    }
    cout << endl;
  }

  // Loop on the tree
  // ----------------
  double flike = 0;
  int evtsinlik = 0;
  int evtsoutlik = 0;
  if( MuScleFitUtils::debug>0 ) {
    cout << "SavedPair.size() = " << MuScleFitUtils::SavedPair.size() << endl;
  }
  for( unsigned int nev=0; nev<MuScleFitUtils::SavedPair.size(); ++nev ) {

    recMu1 = &(MuScleFitUtils::SavedPair[nev].first);
    recMu2 = &(MuScleFitUtils::SavedPair[nev].second);

    // Compute original mass
    // ---------------------
    double mass = MuScleFitUtils::invDimuonMass( *recMu1, *recMu2 );    

    // Compute weight and reference mass (from original mass)
    // ------------------------------------------------------
    double weight = MuScleFitUtils::computeWeight(mass, MuScleFitUtils::iev_);
    if( weight!=0. ) {
      // Compute corrected mass (from previous biases) only if we are currently fitting the scale
      // ----------------------------------------------------------------------------------------
      if( MuScleFitUtils::doScaleFit[MuScleFitUtils::loopCounter] ) {
        corrMu1 = MuScleFitUtils::applyScale(*recMu1, xval, -1);
        corrMu2 = MuScleFitUtils::applyScale(*recMu2, xval,  1);
      }
      else {
        corrMu1 = *recMu1;
        corrMu2 = *recMu2;
      }
      double corrMass = MuScleFitUtils::invDimuonMass(corrMu1, corrMu2);
      double Y = (corrMu1+corrMu2).Rapidity();
      if( MuScleFitUtils::debug>19 ) {
	cout << "[MuScleFitUtils-likelihood]: Original/Corrected resonance mass = " << mass
	     << " / " << corrMass << endl;
      }
      
      // Compute mass resolution
      // -----------------------
      double massResol = MuScleFitUtils::massResolution(corrMu1, corrMu2, xval);
      if (MuScleFitUtils::debug>19) 
	cout << "[MuScleFitUtils-likelihood]: Resolution is " << massResol << endl;

      // Compute probability of this mass value including background modeling
      // --------------------------------------------------------------------
      if (MuScleFitUtils::debug>1) cout << "calling massProb inside likelihood function" << endl;
      double prob = MuScleFitUtils::massProb( corrMass, Y, massResol, xval );
      if (MuScleFitUtils::debug>1) cout << "likelihood:massProb = " << prob << endl;

      // Compute likelihood
      // ------------------
      if( prob>0 ) {
	// flike += log(prob*10000)*weight; // NNBB! x10000 to see if we can recover the problem of boundary
	flike += log(prob)*weight;
	evtsinlik += 1;  // NNBB test: see if likelihood per event is smarter (boundary problem)
      } else {
	evtsoutlik += 1;
      }
      if (MuScleFitUtils::debug>19)
	cout << "[MuScleFitUtils-likelihood]: Mass probability = " << prob << endl;
    } // weight!=0

  } // End of loop on tree events

//   // Protection for low statistic. If the likelihood manages to throw out all the signal
//   // events and stays with ~ 10 events in the resonance window it could have a better likelihood
//   // because of ~ uniformly distributed events (a random combination could be good and spoil the fit).
//   // We require that the number of events included in the fit does not change more than 5% in each minuit loop.
//   bool lowStatPenalty = false;
//   if( MuScleFitUtils::minuitLoop_ > 0 ) {
//     double newEventsOutInRatio = double(evtsinlik);
//     // double newEventsOutInRatio = double(evtsoutlik)/double(evtsinlik);
//     double ratio = newEventsOutInRatio/MuScleFitUtils::oldEventsOutInRatio_;
//     MuScleFitUtils::oldEventsOutInRatio_ = newEventsOutInRatio;
//     if( ratio < 0.8 || ratio > 1.2 ) {
//       cout << "Warning: too much change from oldEventsInLikelihood to newEventsInLikelihood, ratio is = " << ratio << endl;
//       cout << "oldEventsInLikelihood = " << MuScleFitUtils::oldEventsOutInRatio_ << ", newEventsInLikelihood = " << newEventsOutInRatio << endl;
//       lowStatPenalty = true;
//     }
//   }

  // It is a product of probabilities, we compare the sqrt_N of them. Thus N becomes a denominator of the logarithm.
  if( evtsinlik != 0 ) {
    if( MuScleFitUtils::rminPtr_ == 0 ) {
      cout << "ERROR: rminPtr_ = " << MuScleFitUtils::rminPtr_ << ", code will crash" << endl;
    }
    double normalizationArg[] = {1/double(evtsinlik)};
    int ierror = 0;
    MuScleFitUtils::rminPtr_->mnexcm("SET ERR", normalizationArg, 1, ierror);
    fval = -2.*flike/double(evtsinlik);
//     if( lowStatPenalty ) {
//       fval *= 100;
//     }
  }
  else {
    cout << "Problem: Events in likelihood = " << evtsinlik << endl;
    fval = 999999999;
  }
  // fval = -2.*flike;
  if (MuScleFitUtils::debug>19)
    cout << "[MuScleFitUtils-likelihood]: End tree loop with likelihood value = " << fval << endl;

//  #ifdef DEBUG
  // cout << "likelihoodInLoop_ = " << MuScleFitUtils::likelihoodInLoop_ << endl;
  if( MuScleFitUtils::minuitLoop_ < 10000 ) {
    if( MuScleFitUtils::likelihoodInLoop_ == 0 ) cout << "likelihoodInLoop_ = 0" << endl;
    else {
      ++MuScleFitUtils::minuitLoop_;
      MuScleFitUtils::likelihoodInLoop_->SetBinContent(MuScleFitUtils::minuitLoop_, fval);
    }
  }
  else cout << "minuitLoop over 10000. Not filling histogram" << endl;

  if( MuScleFitUtils::debug > 0 ) {
    cout << "Events in likelihood = " << evtsinlik << endl;
    cout << "Events out likelihood = " << evtsoutlik << endl;
  }
//  #endif
}

// Mass fitting routine
// --------------------
vector<TGraphErrors*> MuScleFitUtils::fitMass (TH2F* histo) {

  if (MuScleFitUtils::debug>0) cout << "Fitting " << histo->GetName() << endl;

  vector<TGraphErrors *> results;

  // Results of the fit
  // ------------------
  vector<double> Ftop;
  vector<double> Fwidth;
  vector<double> Fmass;
  vector<double> Etop;
  vector<double> Ewidth;
  vector<double> Emass;
  vector<double> Fchi2;
  // X bin center and width
  // ----------------------
  vector<double> Xcenter;
  vector<double> Ex;
  
  // Fit with lorentzian peak
  // ------------------------
  TF1 *fitFcn = new TF1 ("fitFcn", lorentzianPeak, 70, 110, 3);
  fitFcn->SetParameters (100, 3, 91);
  fitFcn->SetParNames ("Ftop", "Fwidth", "Fmass");
  fitFcn->SetLineWidth (2);

  // Fit slices projected along Y from bins in X 
  // -------------------------------------------
  double cont_min = 20;    // Minimum number of entries
  Int_t binx = histo->GetXaxis()->GetNbins();
  // TFile *f= new TFile("prova.root", "recreate");
  // histo->Write();
  for (int i=1; i<=binx; i++) {
    TH1 * histoY = histo->ProjectionY ("", i, i);
    // histoY->Write();
    double cont = histoY->GetEntries();
    if (cont>cont_min) {
      histoY->Fit ("fitFcn", "0", "", 70, 110);
      double *par = fitFcn->GetParameters();
      double *err = fitFcn->GetParErrors();
      
      Ftop.push_back(par[0]);
      Fwidth.push_back(par[1]);
      Fmass.push_back(par[2]);
      Etop.push_back(err[0]);
      Ewidth.push_back(err[1]);
      Emass.push_back(err[2]);
      
      double chi2 = fitFcn->GetChisquare();
      Fchi2.push_back(chi2);
      
      double xx = histo->GetXaxis()->GetBinCenter(i);
      Xcenter.push_back(xx);
      double ex = 0; // FIXME: you can use the bin width
      Ex.push_back(ex); 
    }
  }
  // f->Close();
  
  // Put the fit results in arrays for TGraphErrors
  // ----------------------------------------------
  const int nn = Fmass.size();
  double *x    = new double[nn];
  double *ym   = new double[nn];
  double *e    = new double[nn];
  double *eym  = new double[nn];
  double *yw   = new double[nn];
  double *eyw  = new double[nn];
  double *yc   = new double[nn];
  
  for (int j=0; j<nn; j++) {
    x[j]   = Xcenter[j];
    ym[j]  = Fmass[j];
    eym[j] = Emass[j];
    yw[j]  = Fwidth[j];
    eyw[j] = Ewidth[j];
    yc[j]  = Fchi2[j];
    e[j]   = Ex[j];
  }
  
  // Create TGraphErrors
  // -------------------
  TString name = histo->GetName();
  TGraphErrors *grM = new TGraphErrors (nn, x, ym, e, eym);
  grM->SetTitle (name+"_M");
  grM->SetName (name+"_M");
  TGraphErrors *grW = new TGraphErrors (nn, x, yw, e, eyw);
  grW->SetTitle (name+"_W");
  grW->SetName (name+"_W");
  TGraphErrors *grC = new TGraphErrors (nn, x, yc, e, e);
  grC->SetTitle (name+"_chi2");
  grC->SetName (name+"_chi2");

  // Cleanup
  // -------
  delete x;
  delete ym;
  delete eym;
  delete yw;
  delete eyw;
  delete yc;
  delete e;
  delete fitFcn;

  results.push_back(grM);
  results.push_back(grW);
  results.push_back(grC);
  return results;
}

// Resolution fitting routine
// --------------------------
vector<TGraphErrors*> MuScleFitUtils::fitReso (TH2F* histo) {
  cout << "Fitting " << histo->GetName() << endl;
  vector<TGraphErrors *> results;

  // Results from fit
  // ----------------
  vector<double> maxs;
  vector<double> means;
  vector<double> sigmas;
  vector<double> chi2s;
  vector<double> Emaxs;
  vector<double> Emeans;
  vector<double> Esigmas;

  // X bin center and width
  // ----------------------
  vector<double> Xcenter;
  vector<double> Ex;
  
  // Fit with a gaussian
  // -------------------
  TF1 *fitFcn = new TF1 ("fitFunc", Gaussian, -0.2, 0.2, 3); 
  fitFcn->SetParameters (100, 0, 0.02);
  fitFcn->SetParNames ("max", "mean", "sigma");
  fitFcn->SetLineWidth (2);
  
  // Fit slices projected along Y from bins in X 
  // -------------------------------------------
  double cont_min = 20;    // Minimum number of entries
  Int_t binx = histo->GetXaxis()->GetNbins();
  for (int i=1; i<=binx; i++) {
    TH1 * histoY = histo->ProjectionY ("", i, i);
    double cont = histoY->GetEntries();
    if (cont>cont_min) {
      histoY->Fit ("fitFunc", "0", "", -0.2, 0.2); 
      double *par = fitFcn->GetParameters();
      double *err = fitFcn->GetParErrors();

      maxs.push_back (par[0]);
      means.push_back (par[1]);
      sigmas.push_back (par[2]);
      Emaxs.push_back (err[0]);
      Emeans.push_back (err[1]);
      Esigmas.push_back (err[2]);
   
      double chi2 = fitFcn->GetChisquare();
      chi2s.push_back (chi2);
      
      double xx = histo->GetXaxis()->GetBinCenter(i);
      Xcenter.push_back (xx);
      double ex = 0; // FIXME: you can use the bin width
      Ex.push_back (ex); 
    }
  }
  
  // Put the fit results in arrays for TGraphErrors
  // ----------------------------------------------
  const int nn = means.size();
  double *x    = new double[nn];
  double *ym   = new double[nn];
  double *e    = new double[nn];
  double *eym  = new double[nn];
  double *yw   = new double[nn];
  double *eyw  = new double[nn];
  double *yc   = new double[nn];
  
  for (int j=0; j<nn; j++) {
    x[j]   = Xcenter[j];
    ym[j]  = means[j];
    eym[j] = Emeans[j];
    // yw[j]  = maxs[j];
    // eyw[j] = Emaxs[j];
    yw[j]  = sigmas[j];
    eyw[j] = Esigmas[j];
    yc[j]  = chi2s[j];
    e[j]   = Ex[j];
  }
  
  // Create TGraphErrors
  // -------------------
  TString name = histo->GetName();
  TGraphErrors *grM = new TGraphErrors (nn, x, ym, e, eym);
  grM->SetTitle (name+"_mean");
  grM->SetName (name+"_mean");
  TGraphErrors *grW = new TGraphErrors (nn, x, yw, e, eyw);
  grW->SetTitle (name+"_sigma");
  grW->SetName (name+"_sigma");
  TGraphErrors *grC = new TGraphErrors (nn, x, yc, e, e);
  grC->SetTitle (name+"_chi2");
  grC->SetName (name+"_chi2");

  // Cleanup
  // -------
  delete x;
  delete ym;
  delete eym;
  delete yw;
  delete eyw;
  delete yc;
  delete e;
  delete fitFcn;
  
  results.push_back (grM);
  results.push_back (grW);
  results.push_back (grC);
  return results;
}

// Mass probability - this is a proper definition of probability (NOT USED YET)
// ----------------------------------------------------------------------------
double MuScleFitUtils::massProb2( const double & mass, const int ires, const double & massResol ) {

  // This is a proper definition of the probability to observe a dimuon mass
  // as distant or more to the reference resonance (ResMass[ires],ResGamma[irs]) if the resolution
  // on the dimuon mass is massResol. It is computed by integrating the convolution
  // between Lorentzian(m,gamma) and Gaussian(mass,massResol) spread on the one-sided
  // tail: if mass<ResMass[ires], the integration is performed from 0 to mass; if mass>ResMass[ires],
  // it is performed from mass to infinity (+-5*massResol is used as 0,inf to truncate).
  // -----------------------------------------------------------------------------------
  double P = 0.;

  if (ires<0 || ires>5 ) {
    // No resonance found within given bounds
    // --------------------------------------
    return P;
  }

  if (massResol==0.) 
    return (0.5*ResGamma[ires]/TMath::Pi())/
      ((mass-ResMass[ires])*(mass-ResMass[ires])+.25*ResGamma[ires]*ResGamma[ires]);

  // If x is the measured dimuon mass, L(ResMass[ires],ResGamma[ires]) is the resonance,
  // and G is the gaussian spread, we define P(x) as follows:
  // 
  //                Int[0:x] dt    Int[0-inf] dm' L(m';ResMass[ires],ResGamma[ires]) G(m'-t)
  // x<M --> P(x) = ---------------------------------------------------------------------------------
  //                Int[0:ResMass[ires]] dt Int[0-inf] dm' L(m';ResMass[ires],ResGamma[ires]) G(m'-t)
  // 
  //                Int[x:inf] dt    Int[0-inf] dm' L(m';ResMass[ires],ResGamma[ires]) G(m'-t)
  // x>M --> P(x) = -----------------------------------------------------------------------------------
  //                Int[ResMass[ires]:inf] dt Int[0-inf] dm' L(m';ResMass[ires],ResGamma[ires]) G(m'-t)
  // 
  // ---------------------------------------------------------------------------------------------------------
  GL2->SetParameters (ResGamma[ires],ResMass[ires],massResol);
  if (mass<=ResMass[ires]) {
    P = GL2->Integral (ResMass[ires]-5*ResGamma[ires], 
		       ResMass[ires]+5*ResGamma[ires], mass-5*massResol, mass)/
        GL2->Integral (ResMass[ires]-5*ResGamma[ires], 
		       ResMass[ires]+5*ResGamma[ires], mass-5*massResol, ResMass[ires]);
  } else if (mass>ResMass[ires]) {
    P = GL2->Integral (ResMass[ires]-5*ResGamma[ires], 
		       ResMass[ires]+5*ResGamma[ires], mass, mass+5*massResol)/
        GL2->Integral (ResMass[ires]-5*ResGamma[ires], 
		       ResMass[ires]+5*ResGamma[ires], ResMass[ires], mass+5*massResol);
  } 

  if (debug>9) cout << "Mass, gamma, mref, width, P: " << mass 
		    << " " << ResGamma[ires] << " " << ResMass[ires] << " " << massResol << " " << P << endl;
  return P;
}

// Mass probability for likelihood computation - no-background version (not used anymore)
// --------------------------------------------------------------------------------------
double MuScleFitUtils::massProb( const double & mass, const double & rapidity, const int ires, const double & massResol )
{  
  // This routine computes the likelihood that a given measured mass "measMass" is
  // the result of resonance #ires if the resolution expected for the two muons is massResol
  // ---------------------------------------------------------------------------------------

  double P = 0.;
  
  // Return Lorentz value for zero resolution cases (like MC)
  // --------------------------------------------------------
  if (massResol==0.) {
    if (debug>9) cout <<  "Mass, gamma , mref, width, P: " << mass 
		      << " " << ResGamma[ires] << " " << ResMass[ires]<< " " << massResol
		      << " : used Lorentz P-value" << endl;
    return (0.5*ResGamma[ires]/TMath::Pi())/((mass-ResMass[ires])*(mass-ResMass[ires])+
					     .25*ResGamma[ires]*ResGamma[ires]);
  }

  // NB defined as below, P is not a "probability" but a likelihood that we observe
  // a dimuon mass "mass", given mRef, gamma, and massResol. It is what we need for the
  // fit which finds the best resolution parameters, though. A definition which is
  // more properly a probability is given below (in massProb2()), where the result
  // cannot be used to fit resolution parameters because the fit would always prefer
  // to set the res parameters to the minimum possible value (best resolution), 
  // to have a probability close to one of observing any mass.
  // -------------------------------------------------------------------------------
  // NNBB the following two lines have been replaced with the six following them, 
  // which provide an improvement of a factor 9 in speed of execution at a 
  // negligible price in precision.
  // ----------------------------------------------------------------------------
  // GL->SetParameters(gamma,mRef,mass,massResol);
  // P = GL->Integral(mass-5*massResol, mass+5*massResol);

  Int_t np = 100;
  double * x = new double[np];
  double * w = new double[np];
  GL->SetParameters (ResGamma[ires], ResMass[ires], mass, massResol);
  GL->CalcGaussLegendreSamplingPoints (np, x, w, 0.1e-15);
  P = GL->IntegralFast (np, x, w, ResMass[ires]-10*ResGamma[ires], ResMass[ires]+10*ResGamma[ires]);
  delete[] x;
  delete[] w;

  // If we are too far away we set P to epsilon and forget about this event
  // ----------------------------------------------------------------------
  if (P<1.0e-12) {
    P = 1.0e-12;
    if (debug>9) cout << "Mass, gamma , mref, width, P: " << mass 
		      << " " << ResGamma[ires] << " " << ResMass[ires] << " " << massResol  
		      << ": used epsilon" << endl;   
    return P;
  }

  if (debug>9) cout << "Mass, gamma , mref, width, P: " << mass 
		    << " " << ResGamma[ires] << " " << ResMass[ires] << " " << massResol 
		    << " " << P << endl;
  return P;
}
