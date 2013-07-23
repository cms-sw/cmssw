#ifndef MuScleFitUtils_H
#define MuScleFitUtils_H

/** \class DTHitQualityUtils
 *
 *  Provide basic functionalities useful for MuScleFit
 *
 *  $Date: 2012/12/20 16:09:29 $
 *  $Revision: 1.11 $
 *  \author S. Bolognesi - INFN Torino / T. Dorigo - INFN Padova
 */

#include <CLHEP/Vector/LorentzVector.h>
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
// #include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TGraphErrors.h"
#include "TH2F.h"
#include "TMinuit.h"

#include "MuonAnalysis/MomentumScaleCalibration/interface/CrossSectionHandler.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundHandler.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"

#include <vector>

// #include "Functions.h"
// class biasFunctionBase<std::vector<double> >;
// class scaleFunctionBase<double*>;
template <class T> class biasFunctionBase;
template <class T> class scaleFunctionBase;
class smearFunctionBase;
template <class T> class resolutionFunctionBase;
class backgroundFunctionBase;
class BackgroundHandler;

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
  static std::pair<SimTrack, SimTrack> findBestSimuRes( const std::vector<SimTrack>& simMuons );
  static std::pair<lorentzVector, lorentzVector> findBestRecoRes( const std::vector<reco::LeafCandidate>& muons );
  static std::pair <lorentzVector, lorentzVector> findGenMuFromRes( const reco::GenParticleCollection* genParticles);
  static std::pair<lorentzVector, lorentzVector> findGenMuFromRes( const edm::HepMCProduct* evtMC );
  static std::pair<lorentzVector, lorentzVector> findSimMuFromRes( const edm::Handle<edm::HepMCProduct> & evtMC,
								   const edm::Handle<edm::SimTrackContainer> & simTracks);

  static std::vector<TGraphErrors*> fitMass (TH2F* histo);
  static std::vector<TGraphErrors*> fitReso (TH2F* histo);

  static lorentzVector applyScale( const lorentzVector & muon, const std::vector<double> & parval, const int charge );
  static lorentzVector applyScale( const lorentzVector & muon, double* parval, const int charge );
  static lorentzVector applyBias( const lorentzVector & muon, const int charge );
  static lorentzVector applySmearing( const lorentzVector & muon );
  static lorentzVector fromPtEtaPhiToPxPyPz( const double* ptEtaPhiE );

  static void minimizeLikelihood();

  static double invDimuonMass( const lorentzVector & mu1, const lorentzVector & mu2 );
  static double massResolution( const lorentzVector & mu1, const lorentzVector & mu2 );
  static double massResolution( const lorentzVector & mu1, const lorentzVector & mu2, const std::vector<double> & parval );
  static double massResolution( const lorentzVector & mu1, const lorentzVector & mu2, std::auto_ptr<double> parval );
  static double massResolution( const lorentzVector & mu1, const lorentzVector & mu2, double* parval );
  static double massResolution( const lorentzVector& mu1, const lorentzVector& mu2, const ResolutionFunction & resolFunc );

  static double massProb( const double & mass, const double & rapidity, const int ires, const double & massResol );
  /* static double massProb( const double & mass, const double & resEta, const double & rapidity, const double & massResol, const std::vector<double> & parval, const bool doUseBkgrWindow = false ); */
  /* static double massProb( const double & mass, const double & resEta, const double & rapidity, const double & massResol, double * parval, const bool doUseBkgrWindow = false ); */
  static double massProb( const double & mass, const double & resEta, const double & rapidity, const double & massResol, const std::vector<double> & parval, const bool doUseBkgrWindow, const double & eta1, const double & eta2 );
  static double massProb( const double & mass, const double & resEta, const double & rapidity, const double & massResol, double * parval, const bool doUseBkgrWindow, const double & eta1, const double & eta2 );
  static double computeWeight( const double & mass, const int iev, const bool doUseBkgrWindow = false );

  static double deltaPhi( const double & phi1, const double & phi2 )
  {
    double deltaPhi = phi1 - phi2;
    while(deltaPhi >= TMath::Pi()) deltaPhi -= 2*TMath::Pi();
    while(deltaPhi < -TMath::Pi()) deltaPhi += 2*TMath::Pi();
    return fabs(deltaPhi);
  }
  /// Without fabs at the end, used to have a symmetric distribution for the resolution fits and variance computations
  static double deltaPhiNoFabs( const double & phi1, const double & phi2 )
  {
    double deltaPhi = phi1 - phi2;
    while(deltaPhi >= TMath::Pi()) deltaPhi -= 2*TMath::Pi();
    while(deltaPhi < -TMath::Pi()) deltaPhi += 2*TMath::Pi();
    return deltaPhi;
  }
  static double deltaR(const double & eta1, const double & eta2, const double & phi1, const double & phi2)
  {
    return sqrt( std::pow( eta1-eta2, 2 ) + std::pow( deltaPhi(phi1, phi2), 2 ) );
  }

  static int debug;       // debug option set by MuScleFit
  static bool ResFound;   // bool flag true if best resonance found (cuts on pt and eta)

  static const int totalResNum; // Total number of resonance: 6
  static double massWindowHalfWidth[3][6]; // parameter set by MuScleFitUtils
  static double ResGamma[6];     // parameter set by MuScleFitUtils
  static double ResMass[6];      // parameter set by MuScleFitUtils
  static double ResMinMass[6];      // parameter set by MuScleFitBase
  static double crossSection[6];
  static const double mMu2;
  static const double muMass;

  // Array of the pdgId of resonances
  static const unsigned int motherPdgIdArray[6];

  static unsigned int loopCounter; // parameter set by MuScleFit

  static int SmearType;
  static smearFunctionBase * smearFunction;
  static int BiasType;
  // No error, we take functions from the same group for scale and bias.
  static scaleFunctionBase<std::vector<double> > * biasFunction;
  static int ResolFitType;
  static resolutionFunctionBase<double *> * resolutionFunction;
  static resolutionFunctionBase<std::vector<double> > * resolutionFunctionForVec;
  static int ScaleFitType;
  static scaleFunctionBase<double*> * scaleFunction;
  static scaleFunctionBase<std::vector<double> > * scaleFunctionForVec;
  static int BgrFitType;
  // Three background regions:
  // - one for the Z
  // - one for the Upsilons
  // - one for J/Psi and Psi2S
  static const int backgroundFunctionsRegions;
  // static backgroundFunctionBase * backgroundFunctionForRegion[];
  // A background function for each resonance
  // static backgroundFunctionBase * backgroundFunction[];

  // The Cross section handler takes care of computing the relative cross
  // sections to be used depending on the resonances that are being fitted.
  // This corresponds to a normalization of the signal pdf.
  static CrossSectionHandler * crossSectionHandler;

  // The background handler takes care of using the correct function in each
  // window, use regions or resonance windows and rescale the fractions when needed
  static BackgroundHandler * backgroundHandler;

  // Parameters used to select whether to do a fit
  static std::vector<int> doResolFit;
  static std::vector<int> doScaleFit;
  static std::vector<int> doCrossSectionFit;
  static std::vector<int> doBackgroundFit;

  static int minuitLoop_;
  static TH1D* likelihoodInLoop_;
  static TH1D* signalProb_;
  static TH1D* backgroundProb_;

  static bool duringMinos_;

  static std::vector<double> parSmear;
  static std::vector<double> parBias;
  static std::vector<double> parResol;
  static std::vector<double> parResolStep;
  static std::vector<double> parResolMin;
  static std::vector<double> parResolMax;
  static std::vector<double> parScale;
  static std::vector<double> parScaleStep;
  static std::vector<double> parScaleMin;
  static std::vector<double> parScaleMax;
  static std::vector<double> parCrossSection;
  static std::vector<double> parBgr;
  static std::vector<int> parResolFix;
  static std::vector<int> parScaleFix;
  static std::vector<int> parCrossSectionFix;
  static std::vector<int> parBgrFix;
  static std::vector<int> parResolOrder;
  static std::vector<int> parScaleOrder;
  static std::vector<int> parCrossSectionOrder;
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
  static double ResMaxSigma[6];         // max sigma of matrix
  static double ResHalfWidth[6];        // halfwidth in matrix
  static int nbins;                     // number of bins in matrix
  static int MuonType; // 0, 1, 2 - 0 is GM, 1 is SM, 2 is track
  static int MuonTypeForCheckMassWindow; // Reduced to be 0, 1 or 2. It is = MuonType when MuonType < 3, = 2 otherwise.

  static std::vector<std::vector<double> > parvalue;
  // static std::map<unsigned int,std::vector<double> > parvalue;
  static std::vector<int> parfix;
  static std::vector<int> parorder;

  static std::vector<std::pair<lorentzVector,lorentzVector> > SavedPair;
  static std::vector<std::pair<lorentzVector,lorentzVector> > ReducedSavedPair;
  static std::vector<std::pair<lorentzVector,lorentzVector> > genPair;
  static std::vector<std::pair<lorentzVector,lorentzVector> > simPair;

  static bool scaleFitNotDone_;

  static bool normalizeLikelihoodByEventNumber_;
  // Pointer to the minuit object
  static TMinuit * rminPtr_;
  // Value stored to check whether to apply a new normalization to the likelihood
  static double oldNormalization_;
  static unsigned int normalizationChanged_;

  // This must be set to true if using events generated with Sherpa
  static bool sherpa_;

  // Decide whether to use the rapidity bins for the Z
  static bool rapidityBinsForZ_;

  static int iev_;

  static bool useProbsFile_;

  // Cuts on the muons to use in the fit
  static bool separateRanges_;
  static double minMuonPt_;
  static double maxMuonPt_;
  static double minMuonEtaFirstRange_;
  static double maxMuonEtaFirstRange_;
  static double minMuonEtaSecondRange_;
  static double maxMuonEtaSecondRange_;
  static double deltaPhiMinCut_;
  static double deltaPhiMaxCut_;

  static bool debugMassResol_;
  static struct massResolComponentsStruct
  {
    double dmdpt1;
    double dmdpt2;
    double dmdphi1;
    double dmdphi2;
    double dmdcotgth1;
    double dmdcotgth2;
  } massResolComponents;

  // Fit accuracy and debug parameters
  static bool startWithSimplex_;
  static bool computeMinosErrors_;
  static bool minimumShapePlots_;

  /// Method to check if the mass value is within the mass window of the i-th resonance.
  // static bool checkMassWindow( const double & mass, const int ires, const double & resMass, const double & leftFactor = 1., const double & rightFactor = 1. );
  static bool checkMassWindow( const double & mass, const double & leftBorder, const double & rightBorder );

  /// Computes the probability given the mass, mass resolution and the arrays with the probabilities and the normalizations.
  static double probability( const double & mass, const double & massResol,
                             const double GLvalue[][1001][1001], const double GLnorm[][1001],
                             const int iRes, const int iY );

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
