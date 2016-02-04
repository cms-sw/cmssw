#ifndef MEPlotWindow_hh
#define MEPlotWindow_hh

#include <TGFrame.h>

#include "MECanvasHolder.hh"

class MusEcalGUI;
class TRootEmbeddedCanvas;

class MEPlotWindow : public MECanvasHolder
{

private:

  MusEcalGUI* _gui;
  TGTransientFrame  *fMain;
  bool fClose;

  TGHorizontalFrame*   fHFrame; 
  TRootEmbeddedCanvas* fEcanvas;


  TString _name;
  TString _printName;

public:

  MEPlotWindow( const TGWindow *p, MusEcalGUI* main, const char* name, UInt_t w, UInt_t h,		
		const char* str1="",
		const char* str2="",
		const char* str3="",
		const char* str4=""
		);
  virtual ~MEPlotWindow();

  void setPrintName( TString printName );
  TString name() { return _name; }
  void write();

  virtual void setPxAndPy( int px, int py );

  // slots
  void CloseWindow();
  void DoClose();
  
ClassDef(MEPlotWindow,0) // MEPlotWindow -- 
    
};

#endif
