#ifndef _TP_GAUSSIAN_LINE_SHAPE_
#define _TP_GAUSSIAN_LINE_SHAPE_

namespace edm{
  class ParameterSet;
}

class RooRealVar;
class RooAddPdf;
class RooGaussian;

class GaussianLineShape{
public:
  GaussianLineShape();
  GaussianLineShape(const edm::ParameterSet& GaussianPSet, RooRealVar *massBins);
  ~GaussianLineShape();

  void Configure(const edm::ParameterSet& GaussianPSet, RooRealVar *massBins);
  void CreatePDF(RooAddPdf *&signalShapePdf);

private:

  void CleanUp();

  RooRealVar  *rooGaussMean_;
  RooRealVar  *rooGaussSigma_;
  RooRealVar  *rooGaussDummyFrac_;
  
  RooGaussian *rooGaussPdf_;

  RooAddPdf *GaussPDF_;
};

#endif
