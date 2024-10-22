#ifndef CalibCalorimetry_EcalLaserAnalyzer_MEGeom_h
#define CalibCalorimetry_EcalLaserAnalyzer_MEGeom_h

//
// Authors  : Gautier Hamel de Monchenault and Julie Malcles, Saclay
//
#include <vector>

#include "MEEBGeom.h"
#include "MEEEGeom.h"

#include "TH2.h"
#include "TCanvas.h"
#include "TGraph.h"

class MEChannel;

class MEGeom {
  // static functions
public:
  // histograms and boundaries
  static TH2* getHist(int ilmr, int unit);

  static TGraph* getBoundary(int ilmr, int unit);
  static void drawHist(int ilmr, int unit, TCanvas* canv = nullptr);

  // global 2D histogram
  static TH2* getGlobalHist(const char* name = nullptr);
  static void setBinGlobalHist(TH2* h, int ix, int iy, int iz, float val);
  static void drawGlobalBoundaries(int lineColor);

  virtual ~MEGeom() {}

private:
  static const int _nbuf;
  static const int _nbinx;
  static const int _nbiny;
  static const float _xmin;
  static const float _xmax;
  static const float _ymin;
  static const float _ymax;
  static const TH2* _h;

  //GHM  ClassDef(MEGeom,0) // MEGeom -- Main geometry class
};

#endif
