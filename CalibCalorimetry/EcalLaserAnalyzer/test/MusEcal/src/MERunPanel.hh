#ifndef MERunPanel_hh
#define MERunPanel_hh

#include <TGFrame.h>
#include <TGButton.h>
#include <TGMsgBox.h>
#include <TGListBox.h>
#include <TGText.h>
#include <TGNumberEntry.h>

class MusEcalGUI;

class MERunPanel {

private:

  TGTransientFrame  *fMain;
  TGVerticalFrame   *fVframe1;
  TGGroupFrame      *f_Run_Group;
  TGLayoutHints     *fHint1, *fHint2, *fHint3;
  TGListBox         *f_RunList;

  TGGroupFrame* f_Range_Group;

  Bool_t fClose;

  MusEcalGUI* _gui;

public:

  MERunPanel( const TGWindow *p, MusEcalGUI* main, UInt_t w, UInt_t h );
  virtual ~MERunPanel();

  // slots
  void CloseWindow();
  void DoClose();
  void SetCurrentRun( UInt_t );

ClassDef(MERunPanel,0) // MERunPanel -- 

};

#endif
