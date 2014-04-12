#include <iostream>
using namespace std;

#include "MEPlotWindow.hh"

#include "MusEcalGUI.hh"
#include "MEClickableCanvas.hh"
#include <TMath.h>
#include <TRootEmbeddedCanvas.h>

ClassImp(MEPlotWindow)

MEPlotWindow::MEPlotWindow( const TGWindow *p, MusEcalGUI* main, const char* name, UInt_t w, UInt_t h, 		const char* str1,
			    const char* str2,
			    const char* str3,
			    const char* str4
			    )
: _gui( main ), _name(name), _printName(name)
{
  unsigned width  = w;
  unsigned height = h;
  unsigned margin = 2;

  fClose = kTRUE;
  
  fMain = new TGTransientFrame( p, main, width, height );
  fMain->Connect("CloseWindow()", "MEPlotWindow", this, "DoClose()");
  fMain->DontCallClose();
  fMain->SetCleanup(kDeepCleanup);
  
  fMain->ChangeOptions( (fMain->GetOptions() & ~kVerticalFrame) | kHorizontalFrame );
  
  fHFrame  = new TGHorizontalFrame( fMain, width, height );
  
  //  fEcanvas = new TRootEmbeddedCanvas( name, fHFrame, width-margin, height-margin );
  //  fEcanvas = new MEClickableCanvas( name, fHFrame, width-margin, height-margin, main );
  fEcanvas = new MEClickableCanvas( name, fHFrame, width-margin, height-margin, this );
  
  TGLayoutHints* fL = new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY,5,5,5,0);
  fHFrame->AddFrame( fEcanvas, fL );
  fMain->AddFrame( fHFrame );
  
  fMain->SetWindowName( name );
  TGDimension size = fMain->GetDefaultSize();
  fMain->Resize(size);

  fMain->SetWMPosition(100,100);
   
  fMain->MapSubwindows();
  fMain->MapWindow();

  SetCanvas( fEcanvas->GetCanvas(), str1, str2, str3, str4 );
}

void 
MEPlotWindow::setPrintName( TString printName )
{
  _printName = printName;
}

void
MEPlotWindow::write()
{
  TString fn_ = _printName;
  // fn_ += ".ps";
  fPad->Print( fn_ );
}

MEPlotWindow::~MEPlotWindow()
{
  _gui->_window.erase( _name );

  fMain->DeleteWindow();   

  //  _gui->setPad();
  //  _gui->_curPad = _gui->fPad;

}

void 
MEPlotWindow::CloseWindow()
{
  delete this;
}

void 
MEPlotWindow::DoClose()
{
  if (fClose)
    CloseWindow();
  else {
    fClose = kTRUE;
    TTimer::SingleShot(150, "MEPlotWindow", this, "CloseWindow()");
  }
}

void 
MEPlotWindow::setPxAndPy( int px, int py ) 
{ 
  MECanvasHolder::setPxAndPy( px, py );
  _gui->windowClicked( this );
}

// XtoPad does not convert user coordinates to NDC !! (it takes care only of lin/log user coordinates 
// to pad coordinates. If linear coordinates   double x = pad->XtoPad(y); returns y otherwise it returns log(y). To convert from user coordinates to NDC, to, eg 

//    Double_t dpx  = gPad->GetX2() - gPad->GetX1();
//    Double_t dpy  = gPad->GetY2() - gPad->GetY1();
//    Double_t xp1  = gPad->GetX1();
//    Double_t yp1  = gPad->GetY1();


//   Double_t xndc = (x-xp1)/dpx 
//   Double_t yndc = (y-yp1)/dpy 
