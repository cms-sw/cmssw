#ifndef PhysicsTools_SideBandSubtraction_h
#define PhysicsTools_SideBandSubtraction_h
#include <string>
#include "TROOT.h"
#include "RooRealVar.h"

class TH1F;
class TF1;
class RooAbsPdf;
class RooDataSet;
class RooFitResult;

typedef struct 
{
  Double_t min;
  Double_t max;
  std::string RegionName;
} SbsRegion;

class SideBandSubtract 
{
 private:
  void print_plot(RooRealVar* printVar, std::string outname);
  Double_t getYield(const std::vector<SbsRegion>& Regions, RooAbsPdf *PDF);
  RooAbsPdf *BackgroundPDF;
  RooAbsPdf *ModelPDF;
  RooDataSet* Data;
  RooRealVar* SeparationVariable;
  bool verbose;
  std::vector<SbsRegion> SignalRegions;
  std::vector<SbsRegion> SideBandRegions;
  std::vector<TH1F> SideBandHistos;
  std::vector<TH1F> RawHistos;
  std::vector<TH1F> SBSHistos;
  std::vector<TH1F*> BaseHistos;
  TH1F* base_histo;
  RooFitResult *fit_result;
  Double_t SignalSidebandRatio;
 public:
  SideBandSubtract();
  /*
  SideBandSubtract(RooAbsPdf *model_shape, 
		   RooAbsPdf *bkg_shape, 
		   RooDataSet* data,
		   RooRealVar* sep_var,
		   bool verb);
  */
  SideBandSubtract(RooAbsPdf *model_shape, 
		   RooAbsPdf *bkg_shape, 
		   RooDataSet* data, 
		   RooRealVar* sep_var, 
		   const std::vector<TH1F*>& base, 
		   bool verb);
  ~SideBandSubtract();
  void addSignalRegion(Double_t min, Double_t max);
  void addSideBandRegion(Double_t min, Double_t max);
  int doGlobalFit();
  int doSubtraction(RooRealVar* variable,Double_t stsratio,Int_t index); //stsratio -> signal to sideband ratio
  void doFastSubtraction(TH1F &Total, TH1F &Result, SbsRegion& leftRegion, SbsRegion& rightRegion);
  void printResults(std::string prefix="");
  void saveResults(std::string outname);
  //the user may want to change the dataset pointer so they can do
  //various subtractions on subsets of the original dataset...
  void setDataSet(RooDataSet* newData);
  RooFitResult* getFitResult();
  std::vector<TH1F> getRawHistos();
  std::vector<TH1F> getSBSHistos();
  std::vector<TH1F*> getBaseHistos();
  Double_t getSTSRatio(); //returns signal-to-sideband ratio
  void resetSBSProducts(); //empties histograms 
};

#endif
