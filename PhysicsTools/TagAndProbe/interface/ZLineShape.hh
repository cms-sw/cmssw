#ifndef _Z_LINE_SHAPE_HH_
#define _Z_LINE_SHAPE_HH_

namespace edm{
  class ParameterSet;
}

class RooRealVar;
class RooAddPdf;
class RooVoigtian;
class RooBifurGauss;
class RooGaussian;

class ZLineShape{
public:
  ZLineShape();
  ZLineShape(const edm::ParameterSet  &pSet,
	     RooRealVar *massBins);
  ~ZLineShape();

  void Configure(const edm::ParameterSet &pSet,
		 RooRealVar *massBins);
  void CreatePDF(RooAddPdf *&ZPDF);

private:
  void CleanUp();
  
  // Private variables/functions needed for ZLineShape
  RooRealVar *rooZMean_;
  RooRealVar *rooZWidth_;
  RooRealVar *rooZSigma_;
  RooRealVar *rooZWidthL_;
  RooRealVar *rooZWidthR_;
  RooRealVar *rooZBifurGaussFrac_;

  RooVoigtian   *rooZVoigtPdf_;
  RooBifurGauss *rooZBifurGaussPdf_;  

  RooAddPdf * ZPDF_;
};

#endif
