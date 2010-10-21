#ifndef MEMultiVarPanel_hh
#define MEMultiVarPanel_hh

#include <TGFrame.h>
#include <TGButton.h>
#include <TGMsgBox.h>
#include <TGListBox.h>
#include <TGText.h>
#include <TGNumberEntry.h>
//#include <TGSlider.h>
#include <TGComboBox.h>

#include <vector>

class MusEcalGUI;

class MEMultiVarPanel {

private:

  TGTransientFrame  *fMain;

  TGHorizontalFrame   *fHframe1;
  TGVerticalFrame   *fVframe1;
  TGVerticalFrame   *fVframe2;
  
  TGLayoutHints* fHint1;
  TGLayoutHints* fHint2;
  TGLayoutHints* fHint3;
  TGLayoutHints* fHint4;
  TGLayoutHints* fHint5;

  std::vector< TGComboBox* >   f_ComboBox;
  std::vector< TGGroupFrame* > f_GroupFrame;

  TGTextButton* f_Go_Button;

  Bool_t fClose;

  MusEcalGUI* _gui;

  int _type;
  int _color;

public:

  MEMultiVarPanel( const TGWindow *p, MusEcalGUI* main, UInt_t w, UInt_t h );
  virtual ~MEMultiVarPanel();

  // slots
  void CloseWindow();
  void DoClose();
  void DoGo();

ClassDef(MEMultiVarPanel,0) // MEMultiVarPanel -- 

};

#endif
