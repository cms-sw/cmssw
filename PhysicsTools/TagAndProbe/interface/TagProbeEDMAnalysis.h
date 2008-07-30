#ifndef PhysicsTools_TagAndProbe_TagProbeEDMAnalysis_h
#define PhysicsTools_TagAndProbe_TagProbeEDMAnalysis_h

//
// Original Author: Nadia Adam (Princeton University) 
//         Created:  Fri May 16 16:48:24 CEST 2008
// $Id: TagProbeEDMAnalysis.h,v 1.1.2.1 2008/07/30 13:29:24 srappocc Exp $
//

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

// ROOT

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>

class TagProbeEDMAnalysis : public edm::EDAnalyzer
{
   public:
      explicit TagProbeEDMAnalysis(const edm::ParameterSet&);
      ~TagProbeEDMAnalysis();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      int SaveHistogram(TH1F& Histo, std::string outFileName, Int_t LogY);

      void SideBandSubtraction(const TH1F& Total, TH1F& Result, Double_t Peak, Double_t SD);
      void ZllEffFitter( std::string &fileName, std::string &bvar, std::vector<double> bins,
			 std::string &bvar2, double bvar2Lo, double bvar2Hi );
      void ZllEffFitter2D( std::string &fileName, std::string &bvar1, std::vector<double> bins1,
			   std::string &bvar2, std::vector<double> bins2 );
      void ZllEffSBS( std::string &fileName, std::string &bvar, std::vector<double> bins,
		      std::string &bvar2, double bvar2Lo, double bvar2Hi );
      void ZllEffSBS2D( std::string &fileName, std::string &bvar1, std::vector<double> bins1,
			std::string &bvar2, std::vector<double> bins2 );

      void ZllEffMCTruth();
      void ZllEffMCTruth2D();

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

      std::vector<double> signalMean_;       // Fit mean
      std::vector<double> signalWidth_;      // Fit width
      std::vector<double> signalSigma_;      // Fit sigma
      std::vector<double> signalWidthL_;     // Fit left width
      std::vector<double> signalWidthR_;     // Fit right width

      std::vector<double> bifurGaussFrac_;   // Fraction of signal shape from bifur Gauss

      std::vector<double> bkgAlpha_;         // Fit background shape alpha
      std::vector<double> bkgBeta_;          // Fit background shape beta
      std::vector<double> bkgPeak_;          // Fit background shape peak
      std::vector<double> bkgGamma_;         // Fit background shape gamma

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

      TH1F *var1Pass_;
      TH1F *var1All_;
      
      TH1F *var2Pass_;
      TH1F *var2All_;

      TH2F *var1var2Pass_;
      TH2F *var1var2All_;

      TH1F* Histograms_;
      int* NumEvents_;
  
      unsigned int numQuantities_;
      bool doAnalyze_;

};

#endif
