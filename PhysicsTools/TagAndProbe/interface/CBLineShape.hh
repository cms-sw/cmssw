#ifndef _CB_LINE_SHAPE_HH_
#define _CB_LINE_SHAPE_HH_

namespace edm {
  class ParameterSet;
}

class RooRealVar;
class RooAddPdf;
class RooCBShape;

class CBLineShape {

public:
  CBLineShape();
  CBLineShape(const edm::ParameterSet &PSet, 
	      RooRealVar *massBins);

  ~CBLineShape();

  void Configure(const edm::ParameterSet &PSet, 
		 RooRealVar *massBins);
  void CreatePDF(RooAddPdf *&CrystalBallPDF);

private:
  void CleanUp();

  RooRealVar *rooCBMean_;
  RooRealVar *rooCBSigma_;
  RooRealVar *rooCBAlpha_;
  RooRealVar *rooCBN_;
  RooRealVar *rooCBDummyFrac_;
  RooCBShape *rooCBPdf_;
  RooAddPdf *CBPDF_;
};

#endif
