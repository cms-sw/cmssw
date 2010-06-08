#ifndef _TP_POLYNOMIAL_LINE_SHAPE_
#define _TP_POLYNOMIAL_LINE_SHAPE_

namespace edm{
  class ParameterSet;
}

class RooRealVar;
class RooPolynomial;
class RooAddPdf;

class PolynomialLineShape {
public:
  PolynomialLineShape();
  PolynomialLineShape(const edm::ParameterSet& polyConfig,
		      RooRealVar *massBins);
  ~PolynomialLineShape();

  void Configure(const edm::ParameterSet& PolynomialConfig,
		 RooRealVar *massBins);
  void CreatePDF(RooAddPdf *&PolyPdf);

private:
  void CleanUp();

  RooRealVar *rooPolyBkgC0_;
  RooRealVar *rooPolyBkgC1_;
  RooRealVar *rooPolyBkgC2_;
  RooRealVar *rooPolyBkgC3_;
  RooRealVar *rooPolyBkgC4_;
  RooRealVar *rooPolyBkgDummyFrac_;
  
  RooPolynomial *rooPolyBkgPdf_;

  RooAddPdf* PolyPDF_;
};

#endif
