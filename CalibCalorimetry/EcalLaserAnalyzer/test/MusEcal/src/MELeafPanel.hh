#ifndef MELeafPanel_hh
#define MELeafPanel_hh

#include <TGFrame.h>
#include <TGButton.h>
#include <TGListBox.h>

#include <vector>

class MusEcalGUI;

class MELeafPanel {

private:

  TGTransientFrame  *fMain;
  
  TGVerticalFrame   *fVframe1;
  TGHorizontalFrame *fHframe1;
  TGHorizontalFrame *fHframe2;
  
  TGListBox*        fVarBox;
  TGListBox*        fZoomBox;
  TGTextButton*     fPlotButton;
  TGTextButton*     fDiffPlotButton;
  TGTextButton*     fOneLevelUpButton;
  
  TGLayoutHints* fHint1;
  TGLayoutHints* fHint2;
  TGLayoutHints* fHint3;
  TGLayoutHints* fHint4;
  TGLayoutHints* fHint5;
  
  MusEcalGUI* _gui;

  int _type;
  int _color;
  int _var;
  int _zoom;

public:

  MELeafPanel( const TGWindow *p, MusEcalGUI* main, UInt_t w, UInt_t h );
  virtual ~MELeafPanel();

  // slots
  void CloseWindow();
  void DoClose();
  void DoPlot();
  void DoDiffPlot();
  void DoOneLevelUp();
  void SetVar(int);
  void SetZoom(int);

ClassDef(MELeafPanel,0) // MELeafPanel -- 

};

#endif
