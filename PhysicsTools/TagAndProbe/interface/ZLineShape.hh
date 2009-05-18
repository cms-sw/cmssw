#ifndef _Z_LINE_SHAPE_HH_
#define _Z_LINE_SHAPE_HH_

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <vector>

class RooRealVar;
class RooAddPdf;
class RooVoigtian;
class RooBifurGauss;
class RooGaussian;

class ZLineShape{
public:
  ZLineShape();
  ZLineShape(const edm::ParameterSet  &pSet);
  ~ZLineShape();

  void Configure(const edm::ParameterSet &pSet);
  void CreatePdf(RooAddPdf *&ZPDF, RooRealVar *rooMass);

private:
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
};

#endif
