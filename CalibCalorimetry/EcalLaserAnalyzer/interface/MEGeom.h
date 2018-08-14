#ifndef MEGeom_hh
#define MEGeom_hh

//
// Authors  : Gautier Hamel de Monchenault and Julie Malcles, Saclay 
//
#include <vector>

#include "MEEBGeom.h"
#include "MEEEGeom.h"

#include <TH2.h>
#include <TCanvas.h>
#include <TGraph.h>

class MEChannel;

class MEGeom
{
  // static functions
public:

  // histograms and boundaries
  static TH2*        getHist( int ilmr, int unit );

  static TGraph* getBoundary( int ilmr, int unit );
  static void drawHist( int ilmr, int unit, TCanvas* canv=nullptr );

  // global 2D histogram
  static TH2* getGlobalHist( const char* name=nullptr );
  static void setBinGlobalHist( TH2* h, 
				int ix, int iy, int iz, float val );  
  static void drawGlobalBoundaries( int lineColor );

  virtual ~MEGeom() {}
  
private:

  static int _nbuf; // 
  static int _nbinx; 
  static int _nbiny; 
  static float _xmin;
  static float _xmax;
  static float _ymin;
  static float _ymax;
  static TH2* _h;

  //GHM  ClassDef(MEGeom,0) // MEGeom -- Main geometry class
};

#endif

