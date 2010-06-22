#ifndef _TP_CMS_BKG_LINE_SHAPE_
#define _TP_CMS_BKG_LINE_SHAPE_

namespace edm{
  class ParameterSet;
}

class RooRealVar;
class RooCMSShapePdf;
class RooAddPdf;

class CMSBkgLineShape {
public:
  
  CMSBkgLineShape();
  CMSBkgLineShape(const edm::ParameterSet  &pSet, RooRealVar *bins);
  ~CMSBkgLineShape();

  void Configure(const edm::ParameterSet &pSet, RooRealVar *bins);
  void CreatePDF(RooAddPdf *&CMSBkgPDF);

private:
  void CleanUp();

  RooRealVar *rooCMSBkgAlpha_;
  RooRealVar *rooCMSBkgBeta_;
  RooRealVar *rooCMSBkgPeak_;
  RooRealVar *rooCMSBkgGamma_;
  RooRealVar *rooCMSBkgDummyFrac_;
  
  RooCMSShapePdf *rooCMSBkgPdf_;

  RooAddPdf *CMSBkgPDF_;
};

#endif
