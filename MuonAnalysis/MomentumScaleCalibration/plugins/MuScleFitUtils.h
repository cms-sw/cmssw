#ifndef MuScleFitUtils_H
#define MuScleFitUtils_H

/** \class DTHitQualityUtils
 *  
 *  Provide basic functionalities useful for MuScleFit
 *
 *  $Date: 2008/12/23 14:20:30 $
 *  $Revision: 1.7 $
 *  \author S. Bolognesi - INFN Torino / T. Dorigo - INFN Padova
 */

#include <CLHEP/Vector/LorentzVector.h>
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <DataFormats/FWLite/interface/Handle.h>
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "TGraphErrors.h"
#include "TH2F.h"

#include <vector>

using namespace std;

// #include "Functions.h"
// class biasFunctionBase<vector<double> >;
// class scaleFunctionBase<double*>;
template <class T> class biasFunctionBase;
template <class T> class scaleFunctionBase;
class smearFunctionBase;
template <class T> class resolutionFunctionBase;
class backgroundFunctionBase;

class SimTrack; 
class TString;
class TTree;

typedef reco::Particle::LorentzVector lorentzVector;

class MuScleFitUtils {
public:
  
  // Constructor
  // ----------
  MuScleFitUtils() {};

  // Destructor
  // ----------
  virtual ~MuScleFitUtils() {};

  // Operations
  // ----------
  static std::pair<SimTrack,SimTrack> findBestSimuRes (std::vector<SimTrack>& simMuons);
  static std::pair<lorentzVector,lorentzVector> findBestRecoRes (std::vector<reco::LeafCandidate>& muons);
  static std::pair <lorentzVector, lorentzVector> findGenMuFromRes(edm::Handle<edm::HepMCProduct> evtMC);
  static std::pair <lorentzVector, lorentzVector> findSimMuFromRes(edm::Handle<edm::HepMCProduct> evtMC, 
								   edm::Handle<edm::SimTrackContainer> simTracks);

  static std::vector<TGraphErrors*> fitMass (TH2F* histo);
  static std::vector<TGraphErrors*> fitReso (TH2F* histo);

  static void cleanEstimator();
  static void computeEstimator (lorentzVector& recMu1, lorentzVector& recMu2, double Zmass);
  static void computeEstimator (lorentzVector& recMu, double Zmass);
  static void returnEstimator();
  
  static lorentzVector applyScale (const lorentzVector &muon, std::vector<double> parval, int charge);
  static lorentzVector applyScale (const lorentzVector &muon, std::auto_ptr<double> parval, int charge);
  static lorentzVector applyScale (const lorentzVector &muon, double* parval, int charge);
  static lorentzVector applyBias (const lorentzVector &muon, int charge);
  static lorentzVector applySmearing (const lorentzVector &muon);
  static lorentzVector fromPtEtaPhiToPxPyPz (double* ptEtaPhiE);

  static void setLikeParameters (double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname);
  static void minimizeLikelihood ();

  static double invDimuonMass (lorentzVector& mu1, lorentzVector& mu2);
  static double massResolution (const lorentzVector& mu1, const lorentzVector& mu2);
  static double massResolution (const lorentzVector& mu1, const lorentzVector& mu2, std::vector<double> parval);
  static double massResolution (const lorentzVector& mu1, const lorentzVector& mu2, std::auto_ptr<double> parval);
  static double massResolution (const lorentzVector& mu1, const lorentzVector& mu2, double* parval);

  static double massProb (double mass, double rapidity, int ires, double massResol);
  static double massProb (double mass, double rapidity, double massResol, std::vector<double> parval);
  static double massProb (double mass, double rapidity, double massResol, std::auto_ptr<double> parval);
  static double massProb (double mass, double rapidity, double massResol, double* parval);
  static double massProb2 (double mass, int ires, double massResol); // Not used yet
  static double computeWeight (double mass);

  static double deltaPhi(double phi1, double phi2) {
    double deltaPhi = phi1 - phi2;
    while(deltaPhi >= TMath::Pi()) deltaPhi -= 2*TMath::Pi();
    while(deltaPhi < -TMath::Pi()) deltaPhi += 2*TMath::Pi();
    return fabs(deltaPhi);
  }
  /// Without fabs at the end, used to have a symmetric distribution for the resolution fits and variance computations
  static double deltaPhiNoFabs(double phi1, double phi2) {
    double deltaPhi = phi1 - phi2;
    while(deltaPhi >= TMath::Pi()) deltaPhi -= 2*TMath::Pi();
    while(deltaPhi < -TMath::Pi()) deltaPhi += 2*TMath::Pi();
    return deltaPhi;
  }
  static double deltaR(const double & eta1, const double & eta2, const double & phi1, const double & phi2) {
    return sqrt( pow( eta1-eta2, 2 ) + pow( deltaPhi(phi1, phi2), 2 ) );
  }

  static int debug;       // debug option set by MuScleFit
  static bool ResFound;   // bool flag true if best resonance found (cuts on pt and eta)

  static double ResHalfWidth[6][3]; // parameter set by MuScleFitUtils
  static double ResGamma[6];     // parameter set by MuScleFitUtils
  static double ResMass[6];      // parameter set by MuScleFitUtils
  static const double mMu2;
  static const double muMass;

  static unsigned int loopCounter; // parameter set by MuScleFit

  static int SmearType;
  static smearFunctionBase * smearFunction;
  static int BiasType;
  // No error, we take functions from the same group for scale and bias.
  static scaleFunctionBase<vector<double> > * biasFunction;
  static int ResolFitType;
  static resolutionFunctionBase<double *> * resolutionFunction;
  static resolutionFunctionBase<vector<double> > * resolutionFunctionForVec;
  static int ScaleFitType;
  static scaleFunctionBase<double*> * scaleFunction;
  static scaleFunctionBase<vector<double> > * scaleFunctionForVec;
  static int BgrFitType;
  static backgroundFunctionBase * backgroundFunction;

  // Parameters used to select whether to do a fit
  static std::vector<int> doResolFit;
  static std::vector<int> doScaleFit;
  static std::vector<int> doBackgroundFit;

  static int minuitLoop_;
  static TH1F* likelihoodInLoop_;

  static std::vector<double> parSmear;
  static std::vector<double> parBias;
  static std::vector<double> parResol;
  static std::vector<double> parScale;
  static std::vector<double> parBgr;
  static std::vector<int> parResolFix;
  static std::vector<int> parScaleFix;
  static std::vector<int> parBgrFix;
  static std::vector<int> parResolOrder;
  static std::vector<int> parScaleOrder;
  static std::vector<int> parBgrOrder;
  static std::vector<int> resfind;
  static int FitStrategy;
  static bool speedup;       // parameter set by MuScleFit - whether to speedup processing
  static double x[7][10000]; // smearing values set by MuScleFit constructor
  static int goodmuon;       // number of events with a usable resonance
  static int counter_resprob;// number of times there are resolution problems
  static double GLZValue[40][1001][1001]; // matrix with integral values of Lorentz * Gaussian 
  static double GLZNorm[40][1001];        // normalization values per each sigma
  static double GLValue[6][1001][1001]; // matrix with integral values of Lorentz * Gaussian 
  static double GLNorm[6][1001];        // normalization values per each sigma
  static double ResMaxSigma[6][3];         // max sigma of matrix
  static int nbins;                     // number of bins in matrix
  static int MuonType; // 0, 1, 2 - 0 is GM, 1 is SM, 2 is track

  static std::vector<std::vector<double> > parvalue;
  // static std::map<unsigned int,std::vector<double> > parvalue;
  static std::vector<int> parfix;
  static std::vector<int> parorder;
  
  static std::vector<std::pair<lorentzVector,lorentzVector> > SavedPair;

protected:

private:
  
   struct byPt {
     bool operator() (const reco::Muon &a, const reco::Muon &b) const { 
       return a.pt() > b.pt();
     }
   };

};

extern "C" void likelihood (int& npar, double* grad, double& fval, double* xval, int flag);

#endif
