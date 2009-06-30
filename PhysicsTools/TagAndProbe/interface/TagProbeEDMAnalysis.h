#ifndef PhysicsTools_TagAndProbe_TagProbeEDMAnalysis_h
#define PhysicsTools_TagAndProbe_TagProbeEDMAnalysis_h

//
// Original Author: Nadia Adam (Princeton University) 
//         Created:  Fri May 16 16:48:24 CEST 2008
// $Id: TagProbeEDMAnalysis.h,v 1.10 2009/02/17 17:27:49 haupt Exp $
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


// system include files
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/TagAndProbe/interface/RooCMSShapePdf.h"

// Used for 2D efficiency reading/writing
#include "PhysicsTools/TagAndProbe/interface/EffTableLoader.h"

// ROOT

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>

// RooFit headers
#include <RooAbsData.h>
#include <RooDataSet.h>
#include <RooAddPdf.h>
#include <RooBifurGauss.h>
#include <RooBreitWigner.h>
#include <RooCategory.h>
#include <RooCatType.h>
#include <RooCBShape.h>
#include <RooChi2Var.h>
#include <RooDataHist.h>
#include <RooFitResult.h>
#include <RooGaussian.h>
#include <RooGenericPdf.h>
#include <RooGlobalFunc.h>
#include <RooLandau.h>
#include <RooMinuit.h>
#include <RooNLLVar.h>
#include <RooPlot.h>
#include <RooPolynomial.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooTreeData.h>
#include <RooVoigtian.h>


class TagProbeEDMAnalysis : public edm::EDAnalyzer
{
   public:
      explicit TagProbeEDMAnalysis(const edm::ParameterSet&);
      ~TagProbeEDMAnalysis();

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      int SaveHistogram(TH1F& Histo, std::string outFileName, Int_t LogY);

      void SideBandSubtraction(const TH1F& Total, TH1F& Result, Double_t Peak, Double_t SD);
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

      void cleanFitVariables()
      {
	 if( !calcEffsFitter_ ) return;

	 if( rooMass_ != NULL )               delete rooMass_;
	 if( signalShapePdf_ != NULL )        delete signalShapePdf_;
	 if( bkgShapePdf_ != NULL )           delete bkgShapePdf_;

	 // Clean Z line shape
	 if( fitZLineShape_ )
	 {
	    if( rooZMean_ != NULL )           delete rooZMean_;
	    if( rooZWidth_ != NULL )          delete rooZWidth_;
	    if( rooZSigma_ != NULL )          delete rooZSigma_;
	    if( rooZWidthL_ != NULL )         delete rooZWidthL_;
	    if( rooZWidthR_ != NULL )         delete rooZWidthR_;
	    if( rooZBifurGaussFrac_ != NULL ) delete rooZBifurGaussFrac_;
	    if( rooZVoigtPdf_ != NULL )       delete rooZVoigtPdf_;
	    if( rooZBifurGaussPdf_ != NULL )  delete rooZBifurGaussPdf_;

	    if( floatFailZMean_ && rooFailZMean_ != NULL )   delete rooFailZMean_;
	    if( floatFailZWidth_ && rooFailZWidth_ != NULL ) delete rooFailZWidth_;
	    if( floatFailZSigma_ && rooFailZSigma_ != NULL ) delete rooFailZSigma_;
	    if( floatFailZWidthL_ && rooFailZWidthL_ != NULL ) delete rooFailZWidthL_;
	    if( floatFailZWidthR_ && rooFailZWidthR_ != NULL ) delete rooFailZWidthR_;
	    if( floatFailZBifurGaussFrac_ && rooFailZBifurGaussFrac_ != NULL ) delete rooFailZBifurGaussFrac_;
	    if( rooFailZVoigtPdf_ != NULL )       delete rooFailZVoigtPdf_;
	    if( rooFailZBifurGaussPdf_ != NULL )  delete rooFailZBifurGaussPdf_;

	    if( signalShapeFailPdf_ != NULL )     delete signalShapeFailPdf_;
	 }

	 if( fitCBLineShape_ )
	 {
	    // Clean CB line shape
	    if( rooCBMean_ != NULL )           delete rooCBMean_;
	    if( rooCBSigma_ != NULL )          delete rooCBSigma_;
	    if( rooCBAlpha_ != NULL )          delete rooCBAlpha_;
	    if( rooCBN_ != NULL )              delete rooCBN_;
	    if( rooCBDummyFrac_ != NULL )      delete rooCBDummyFrac_;
	    if( rooCBPdf_ != NULL )            delete rooCBPdf_;
	 }

	 if( fitGaussLineShape_ )
	 {
	    // Clean Gauss line shape
	    if( rooGaussMean_ != NULL )           delete rooGaussMean_;
	    if( rooGaussSigma_ != NULL )          delete rooGaussSigma_;
	    if( rooGaussDummyFrac_ != NULL )      delete rooGaussDummyFrac_;
	    if( rooGaussPdf_ != NULL )            delete rooGaussPdf_;
	 }

	 if( fitCMSBkgLineShape_ )
	 {
	    // Clean CMS Bkg line shape
	    if( rooCMSBkgAlpha_ != NULL )      delete rooCMSBkgAlpha_;
	    if( rooCMSBkgBeta_ != NULL )       delete rooCMSBkgBeta_;
	    if( rooCMSBkgPeak_ != NULL )       delete rooCMSBkgPeak_;
	    if( rooCMSBkgGamma_ != NULL )      delete rooCMSBkgGamma_;
	    if( rooCMSBkgDummyFrac_ != NULL )  delete rooCMSBkgDummyFrac_;
	    if( rooCMSBkgPdf_ != NULL )        delete rooCMSBkgPdf_;
	 }

	 if( fitPolyBkgLineShape_ )
	 {
	    // Clean the polynomial background line shape
	    if( rooPolyBkgC0_ != NULL )        delete rooPolyBkgC0_;
	    if( rooPolyBkgC1_ != NULL )        delete rooPolyBkgC1_;
	    if( rooPolyBkgC2_ != NULL )        delete rooPolyBkgC2_;
	    if( rooPolyBkgC3_ != NULL )        delete rooPolyBkgC3_;
	    if( rooPolyBkgC4_ != NULL )        delete rooPolyBkgC4_;
	    if( rooPolyBkgDummyFrac_ != NULL ) delete rooPolyBkgDummyFrac_;
	    if( rooPolyBkgPdf_ != NULL )       delete rooPolyBkgPdf_;
	 }
      }

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

};

#endif
