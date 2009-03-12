#ifndef MECanvasHolder_hh
#define MECanvasHolder_hh

#include <TROOT.h>
#include <TVirtualPadEditor.h>
#include <TLatex.h>
#include <TPaveText.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TH1.h>
#include <TH2.h>

class MECanvasHolder
{
public:

  // action after a double click
  virtual void setPxAndPy( int px, int py ); 

  TPad* getPad() { return fPad; }
  void setPad();

  TH1* curHist() { return _h; }
  void setCurHist( TH1* h ) { _h=h; }
  void getCurXY( float& x, float& y ) { x=_x; y=_y; }

  // ROOT style and Histograms
  void setSessionStyle();
  static void setHistoStyle( TH1* );

public:

  // contructors/destructor
  MECanvasHolder();
  virtual  ~MECanvasHolder();

  // Canvas
  TCanvas*    fCanvas;          //Canvas of fECanvas
  TPad*       fPad;
  Int_t       fTopXGen;         //Top x  of general canvas fCanvas
  Int_t       fTopYGen;         //Top y  of general canvas fCanvas
  UInt_t      fWidthGen;        //Width  of general canvas fCanvas
  UInt_t      fHeigthGen;       //Heigth of general canvas fCanvas
  TString     fDate;            //Date when program runs
  TString     fTime;            //Time when program runs


  // Welcome pave (from Monecal)
  TPaveText* fWelcomePave;    
  Bool_t     fWelcomeState;  
  TText*     fWelcomeTitle;    
  TText*     fWelcomeL0;      

  TLatex* fTexTL;
  TLatex* fTexTR;
  TLatex* fTexBL;
  TLatex* fTexBR;

  // current position in the canvas after a double click
  int _px;
  int _py;
  float _x;
  float _y;

  // current histogram
  TH1* _h;

  // Methods inherited from Monecal
  void      SetCanvas( TCanvas* canvas, 
		       const char* str1="", 
		       const char* str2="", 
		       const char* str3="", 
		       const char* str4="" 
		       );
  void      CanvasModified();
  void      ClearWelcome();
  void      ShowWelcome( bool=false );
  void      SetDate();

  void setHessPalette();

private:
  
  double _scale;
  double _refw;

// declare to ROOT dictionary
ClassDef(MECanvasHolder,0) // MECanvasHolder -- Monitoring utility for survey of Ecal
};

#endif

