#ifndef _TPRooSimultaneousFitter_HH_
#define _TPRooSimultaneousFitter_HH_

// STL
#include <vector>
#include <string>
#include <memory>

class RooArgSet;
class RooRealVar;
class RooAddPdf;
class RooSimultaneous;
class RooStringVar;
class RooFitResult;
class TTree;
class TFile;
class RooDataSet;
class RooDataHist;

class TPRooSimultaneousFitter{
public:

  TPRooSimultaneousFitter();


  ~TPRooSimultaneousFitter();

  void configure(RooRealVar &ResonanceMass, 
		 TTree *fitTree,
		 const std::string &bvar1, 
		 const std::vector< double >& bins1,
		 const int bin1,
		 const std::string &bvar2, 
		 const std::vector< double >& bins2,
		 const int bin2,
		 std::vector<double> &efficiency,
		 std::vector<double> &numSignal,
		 std::vector<double> &numBkgPass ,
		 std::vector<double> &numBkgFail);

  void createTotalPDF(RooAddPdf *signalShapePdf,
		      RooAddPdf *bkgShapePdf);

  RooFitResult* performFit(bool UnBinnedFit,
			   int& npassResult, 
			   int& nfailResult);

  void persistFitresultsToRoot(RooFitResult* FitResults);
  

  RooArgSet readFitresultsFromRoot(char* filename);
  
  void saveCanvasTPResults(TFile *outputfile,
			   char* filename,
			   RooAddPdf *signalShapePdf,
			   RooAddPdf *bkgShapePdf,
			   int& npassResult,
			   int& nfailResult,
			   const bool is2D);
  

private:

  std::auto_ptr<RooRealVar> rooEfficiency_;
  std::auto_ptr<RooRealVar> rooNumSignal_;
  std::auto_ptr<RooRealVar> rooNumBkgPass_;
  std::auto_ptr<RooRealVar> rooNumBkgFail_;

  std::auto_ptr<RooSimultaneous> rooTotalPDF_;
  // RooDataSet* rooData_;
  std::auto_ptr<RooDataSet> rooData_;
  std::auto_ptr<RooDataHist> roobData_;

  std::auto_ptr<RooRealVar> rooMass_;
  std::auto_ptr<RooRealVar> rooVar1_;
  std::auto_ptr<RooRealVar> rooVar2_;

  int rooBin1_;
  int rooBin2_;




};
#endif
