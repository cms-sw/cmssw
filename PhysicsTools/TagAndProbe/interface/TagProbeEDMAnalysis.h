#ifndef PhysicsTools_TagAndProbe_TagProbeEDMAnalysis_h
#define PhysicsTools_TagAndProbe_TagProbeEDMAnalysis_h

//
// Original Author: Nadia Adam (Princeton University) 
//         Created:  Fri May 16 16:48:24 CEST 2008
// $Id: TagProbeEDMAnalysis.h,v 1.24 2010/01/06 22:47:40 gpetrucc Exp $
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

#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/Utilities/interface/SideBandSubtraction.h"

class FeldmanCousinsBinomialInterval;

class EffTableLoader;
class SideBandSubtract;

class ZLineShape;
class CBLineShape;
class GaussianLineShape;
class PolynomialLineShape;
class CMSBkgLineShape;

class RooRealVar;
class RooAddPdf;

class TH1F;
//class TFile;
class TTree;
class TH2F;

class TagProbeEDMAnalysis : public edm::EDAnalyzer{

   public:
      explicit TagProbeEDMAnalysis(const edm::ParameterSet&);
      ~TagProbeEDMAnalysis();

   private:
      virtual void beginJob() {}
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      int SaveHistogram(TH1F& Histo, std::string outFileName, int LogY = 0);
      void TPEffFitter( std::string &fileName, std::string &bvar, std::vector<double> bins,
			 std::string &bvar2, double bvar2Lo, double bvar2Hi );

      void TPEffFitter2D (
			  const std::string &fileName, 
			  std::string &bvar1, 
			  std::vector< double > &bins1,
			  const std::string &bvar2, 
			  std::vector<double> &bins2 );

      void TPEffSBS(std::string &fileName, std::string &bvar, std::vector<double> bins,
		      std::string &bvar2, double bvar2Lo, double bvar2Hi );
      void TPEffSBS2D( std::string &fileName, std::string &bvar1, std::vector<double> bins1,
			std::string &bvar2, std::vector<double> bins2 );
      void makeSignalPdf();
      void makeBkgPdf();

      void performFit( const std::string &bvar1, const std::vector< double >& bins1, const int bin1,
		  const std::string &bvar2, const std::vector< double >& bins2, const int bin2,
		  double &eff, double &hierr, double &loerr, double &chi2Val, double& quality, const bool is2D = false );



      void TPEffMCTruth();
      void TPEffMCTruth2D();

      void CalculateEfficiencies();

      void CreateFitTree();
      void FillFitTree(const edm::Event& iEvent);
      void CheckEfficiencyVariables();


      void CreateMCTree();
      void FillMCTree(const edm::Event& iEvent);

      void InitializeMCHistograms();
      void ReadMCHistograms();
      void WriteMCHistograms();

      void cleanFitVariables();

/*       // Histogram drawing input variables */
/*       std::vector<std::string> quantities_;   */
/*       std::vector<std::string> conditions_;   */
/*       std::vector<std::string> outputFileNames_; */

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

      std::string  massName_;
      unsigned int massNbins_;           // Number of bins in the fit
      double massLow_;          // Lower bound for fit range
      double massHigh_;         // Upper bound for fit range
      double inweight_;
      
      std::string var1Name_;          // Name of variable one (default pt)
      std::string var1NameUp_;        // Name of variable one uppercase (default Pt)
      unsigned int var1Nbins_;                 // Number of var1 eff bins
      double var1Low_;                // Lower bound for var1 eff range
      double var1High_;               // Upper bound for var1 eff range
      std::vector<double> var1Bins_;  // Bin boundaries for var1 if non-uniform desired

      std::string var2Name_;         // Name of variable two (default eta)
      std::string var2NameUp_;       // Name of variable two uppercase (default Eta)
      unsigned int var2Nbins_;                // Number of var2 eff bins
      double var2Low_;               // Lower bound for var2 eff range
      double var2High_;              // Upper bound for var2 eff range
      std::vector<double> var2Bins_; // Bin boundaries for var2 if non-uniform desired

      std::string passingProbeName_; // Name of the column telling if the probe passes or not

      bool hasWeights_;              // If the tree has weights
      std::string weightName_;       // Name of the branch holding the weights

      bool doTextDefinedBins_;         // Allow the 2-D bins to be read in from a text file into 1-D regions
      std::string textBinsFile_;       // This is the name of the file that holds the 2D bin information
      EffTableLoader* effBinsFromTxt_; // This holds the efficiency bins information
      // New SideBandSubtract interface and required variables
      SbsRegion leftRegion_;
      SbsRegion rightRegion_;
      SideBandSubtract* SBS_;

      ZLineShape* zLineShape_;
      CBLineShape* cbLineShape_;
      GaussianLineShape* gaussLineShape_;
      PolynomialLineShape* polyBkgLineShape_;
      CMSBkgLineShape* cmsBkgLineShape_;

      // The signal & background Pdf & Fit variable
      RooRealVar *rooMass_;
      RooAddPdf  *signalShapePdf_;
      RooAddPdf  *signalShapeFailPdf_;
      RooAddPdf  *bkgShapePdf_;
      
      std::vector<double> efficiency_;       // Signal efficiency from fit
      std::vector<double> numSignal_;        // Signal events from fit
      std::vector<double> numBkgPass_;       // Background events passing from fit
      std::vector<double> numBkgFail_;       // Background events failing from fit

      std::string mode_;                     // Mode of operation (Normal,Read,Write)  
      std::string fitFileName_;              // Name of the root file to write to
      std::vector<std::string> readFiles_;   // Files to read from ... if mode == READ
      std::string readDirectory_;            // TTrees have to be taken from subdirectory 'readDirectory'

      //      TFile *outRootFile_;
      TTree *fitTree_;
      TTree *mcTree_;
      int    ProbePass_;
      double Mass_;
      double Var1_;
      double Var2_;
      double Weight_;
      double MCVar1_;
      double MCVar2_;

      // True MC Truth
      TH1F *var1Pass_;
      TH1F *var1All_;
      
      TH1F *var2Pass_;
      TH1F *var2All_;

      TH2F *var1var2Pass_;
      TH2F *var1var2All_;

      int* NumEvents_;

      bool doAnalyze_;

      std::auto_ptr<FeldmanCousinsBinomialInterval> FCIntervals;

      edm::Service<TFileService> fs;
      TFileDirectory mcDetails_, fitDetails_, sbsDetails_;
};

#endif
