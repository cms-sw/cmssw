#ifndef PhysicsTools_TagAndProbe_TagProbeEDMAnalysis_h
#define PhysicsTools_TagAndProbe_TagProbeEDMAnalysis_h

//
// Original Author: Nadia Adam (Princeton University) 
//         Created:  Fri May 16 16:48:24 CEST 2008
// $Id: TagProbeEDMAnalysis.h,v 1.12 2009/04/06 19:18:35 ahunt Exp $
//
//
// Kalanand Mishra: July 1, 2008 
// Added a configurable option "useRecoVarsForTruthMatchedCands" 
// (default == 'false') to use reconstructed or detector values 
// (instead of MC generated values) for var1 and var2 when doing MC truth efficiencies.
//
// Kalanand Mishra: October 7, 2008 
// Removed duplication of code in the fitting machinery. 
// Also, fixed the problem with RooDataSet declaration.

#include "FWCore/Framework/interface/EDAnalyzer.h"

class TH1F;
class TFile;
class TTree;
class TH2F;

class EffTableLoader;

class RooRealVar;
class RooAddPdf;
class RooVoigtian;
class RooBifurGauss;
class RooCBShape;
class RooGaussian;
class RooCMSShapePdf;
class RooPolynomial;

class TagProbeEDMAnalysis : public edm::EDAnalyzer
{
   public:
      explicit TagProbeEDMAnalysis(const edm::ParameterSet&);
      ~TagProbeEDMAnalysis();

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      int SaveHistogram(TH1F& Histo, std::string outFileName, int LogY);

      void SideBandSubtraction(const TH1F& Total, TH1F& Result, double Peak, double SD);
      void TPEffFitter( std::string &fileName, std::string &bvar, std::vector<double> bins,
			 std::string &bvar2, double bvar2Lo, double bvar2Hi );
      void TPEffFitter2D( std::string &fileName, std::string &bvar1, std::vector<double> bins1,
			   std::string &bvar2, std::vector<double> bins2 );
      void TPEffSBS( std::string &fileName, std::string &bvar, std::vector<double> bins,
		      std::string &bvar2, double bvar2Lo, double bvar2Hi );
      void TPEffSBS2D( std::string &fileName, std::string &bvar1, std::vector<double> bins1,
			std::string &bvar2, std::vector<double> bins2 );
      void makeSignalPdf();
      void makeBkgPdf();
      void doFit( std::string &bvar1, std::vector< double > bins1, int bin1, 
		  std::string &bvar2, std::vector<double> bins2, int bin2, 
                  double &eff, double &err, bool is2D = false );

      void TPEffMCTruth();
      void TPEffMCTruth2D();

      void CalculateEfficiencies();
      void cleanFitVariables();

      // Histogram drawing input variables
      std::vector<std::string> quantities_;  
      std::vector<std::string> conditions_;  
      std::vector<std::string> outputFileNames_;

      std::vector<unsigned int> XBins_;
      std::vector<double> XMax_;
      std::vector<double> XMin_;
      std::vector<unsigned int> logY_;

      std::vector<double> lumi_;
      std::vector<double> xsection_;
      std::vector<double> weight_;
      bool Verbose_;

      // Fitting/Efficiency calculation inputs
      int tagProbeType_;        // If more than one tag-probe type is stored select

      bool calcEffsSB_;         // Calculate effs using SB subtraction for these TTrees?
      bool calcEffsFitter_;     // Calculate effs using fitter for these TTrees?
      bool calcEffsTruth_;      // Calculate effs using MC truth for these TTrees

      int truthParentId_;       // Set the parent id for the truch calculation, use 23 for 
                                // signal Z, 0 for no parent requirement

      bool unbinnedFit_;        // Do a binned/unbinned fit
      bool do2DFit_;            // Do the 2D fit as well
      bool useRecoVarsForTruthMatchedCands_; // use reco vars for calcEffsTruth

      int massNbins_;           // Number of bins in the fit
      double massLow_;          // Lower bound for fit range
      double massHigh_;         // Upper bound for fit range

      double inweight_;
      
      std::string var1Name_;          // Name of variable one (default pt)
      std::string var1NameUp_;        // Name of variable one uppercase (default Pt)
      int var1Nbins_;                 // Number of var1 eff bins
      double var1Low_;                // Lower bound for var1 eff range
      double var1High_;               // Upper bound for var1 eff range
      std::vector<double> var1Bins_;  // Bin boundaries for var1 if non-uniform desired

      std::string var2Name_;         // Name of variable two (default eta)
      std::string var2NameUp_;       // Name of variable two uppercase (default Eta)
      int var2Nbins_;                // Number of var2 eff bins
      double var2Low_;               // Lower bound for var2 eff range
      double var2High_;              // Upper bound for var2 eff range
      std::vector<double> var2Bins_; // Bin boundaries for var2 if non-uniform desired

      bool doTextDefinedBins_;         // Allow the 2-D bins to be read in from a text file into 1-D regions
      std::string textBinsFile_;       // This is the name of the file that holds the 2D bin information
      EffTableLoader* effBinsFromTxt_; // This holds the efficiency bins information

      // Parameter set fo the available fit functions 

      // The signal & background Pdf & Fit variable
      RooRealVar *rooMass_;
      RooAddPdf  *signalShapePdf_;
      RooAddPdf  *signalShapeFailPdf_;
      RooAddPdf  *bkgShapePdf_;
      
      // 1. Z line shape
      bool fitZLineShape_;
      edm::ParameterSet   ZLineShape_;
      std::vector<double> zMean_;       // Fit mean
      std::vector<double> zWidth_;      // Fit width
      std::vector<double> zSigma_;      // Fit sigma
      std::vector<double> zWidthL_;     // Fit left width
      std::vector<double> zWidthR_;     // Fit right width
      std::vector<double> zBifurGaussFrac_;   // Fraction of signal shape from bifur Gauss

      // Private variables/functions needed for ZLineShape
      RooRealVar *rooZMean_;
      RooRealVar *rooZWidth_;
      RooRealVar *rooZSigma_;
      RooRealVar *rooZWidthL_;
      RooRealVar *rooZWidthR_;
      RooRealVar *rooZBifurGaussFrac_;

      RooVoigtian   *rooZVoigtPdf_;
      RooBifurGauss *rooZBifurGaussPdf_;

      // In case we need the failing probes to float separately
      bool        floatFailZMean_;
      bool        floatFailZWidth_;
      bool        floatFailZSigma_;
      bool        floatFailZWidthL_;
      bool        floatFailZWidthR_;
      bool        floatFailZBifurGaussFrac_;
      RooRealVar *rooFailZMean_;
      RooRealVar *rooFailZWidth_;
      RooRealVar *rooFailZSigma_;
      RooRealVar *rooFailZWidthL_;
      RooRealVar *rooFailZWidthR_;
      RooRealVar *rooFailZBifurGaussFrac_;

      RooVoigtian   *rooFailZVoigtPdf_;
      RooBifurGauss *rooFailZBifurGaussPdf_;

      // 2. Crystal Ball Line Shape
      bool fitCBLineShape_;
      edm::ParameterSet   CBLineShape_;
      std::vector<double> cbMean_;           // Fit mean
      std::vector<double> cbSigma_;          // Fit sigma
      std::vector<double> cbAlpha_;          // Fit alpha
      std::vector<double> cbN_;              // Fit n

      // Private variables/functions needed for CBLineShape
      RooRealVar *rooCBMean_;
      RooRealVar *rooCBSigma_;
      RooRealVar *rooCBAlpha_;
      RooRealVar *rooCBN_;
      RooRealVar *rooCBDummyFrac_;

      RooCBShape *rooCBPdf_;

      // 3. Plain old Gaussian Line Shape
      bool fitGaussLineShape_;
      edm::ParameterSet   GaussLineShape_;
      std::vector<double> gaussMean_;           // Fit mean
      std::vector<double> gaussSigma_;          // Fit sigma

      // Private variables/functions needed for CBLineShape
      RooRealVar  *rooGaussMean_;
      RooRealVar  *rooGaussSigma_;
      RooRealVar  *rooGaussDummyFrac_;

      RooGaussian *rooGaussPdf_;

      // The background Pdf and fit variables

      // 1. CMS Background shape
      bool fitCMSBkgLineShape_;
      edm::ParameterSet   CMSBkgLineShape_;
      std::vector<double> cmsBkgAlpha_;         // Fit background shape alpha
      std::vector<double> cmsBkgBeta_;          // Fit background shape beta
      std::vector<double> cmsBkgPeak_;          // Fit background shape peak
      std::vector<double> cmsBkgGamma_;         // Fit background shape gamma

      RooRealVar *rooCMSBkgAlpha_;
      RooRealVar *rooCMSBkgBeta_;
      RooRealVar *rooCMSBkgPeak_;
      RooRealVar *rooCMSBkgGamma_;
      RooRealVar *rooCMSBkgDummyFrac_;

      RooCMSShapePdf *rooCMSBkgPdf_;

      // 2. Polynomial background shape (up to 4th order)
      bool fitPolyBkgLineShape_;
      edm::ParameterSet PolyBkgLineShape_;
      std::vector<double> polyBkgC0_;
      std::vector<double> polyBkgC1_;
      std::vector<double> polyBkgC2_;
      std::vector<double> polyBkgC3_;
      std::vector<double> polyBkgC4_;

      RooRealVar *rooPolyBkgC0_;
      RooRealVar *rooPolyBkgC1_;
      RooRealVar *rooPolyBkgC2_;
      RooRealVar *rooPolyBkgC3_;
      RooRealVar *rooPolyBkgC4_;
      RooRealVar *rooPolyBkgDummyFrac_;

      RooPolynomial *rooPolyBkgPdf_;


      std::vector<double> efficiency_;       // Signal efficiency from fit
      std::vector<double> numSignal_;        // Signal events from fit
      std::vector<double> numBkgPass_;       // Background events passing from fit
      std::vector<double> numBkgFail_;       // Background events failing from fit

      double SBSPeak_;                       // Sideband sub peak
      double SBSStanDev_;                    // Sideband sub standard deviation

      std::string mode_;                     // Mode of operation (Normal,Read,Write)  
      std::string fitFileName_;              // Name of the root file to write to
      std::vector<std::string> readFiles_;   // Files to read from ... if mode == READ

      TFile *outRootFile_;
      TTree *fitTree_;
      int    ProbePass_;
      double Mass_;
      double Var1_;
      double Var2_;
      double Weight_;

      // True MC Truth
      TH1F *var1Pass_;
      TH1F *var1All_;
      
      TH1F *var2Pass_;
      TH1F *var2All_;

      TH2F *var1var2Pass_;
      TH2F *var1var2All_;

      // Tag-Probe biased MC Truth
      TH1F *var1BiasPass_;
      TH1F *var1BiasAll_;
      
      TH1F *var2BiasPass_;
      TH1F *var2BiasAll_;

      TH2F *var1var2BiasPass_;
      TH2F *var1var2BiasAll_;

      TH1F* Histograms_;

      int* NumEvents_;

      unsigned int numQuantities_;
      bool doAnalyze_;

      std::stringstream roofitstream;
};

#endif
