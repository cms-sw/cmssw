#ifndef MEChanPanel_hh
#define MEChanPanel_hh

#include <TGFrame.h>
#include <TGButton.h>
#include <TGMsgBox.h>
#include <TGListBox.h>
#include <TGText.h>
#include <TGNumberEntry.h>

class MusEcalGUI;

class MEChanPanel {

private:

  TGTransientFrame  *fMain;

  TGHorizontalFrame   *fHframe1;

  TGGroupFrame      *f_Channel_Group;
  TGGroupFrame      *f_Channel_ID_Group;
  TGGroupFrame      *f_Channel_XY_Group;

  TGHorizontalFrame *f_X_Group, *f_Y_Group;
  TGNumberEntry     *f_X, *f_Y;
  TGTextButton      *f_XY_Button;
  TGLabel           *f_X_Label, *f_Y_Label;

  TGHorizontalFrame *f_ID_Group;
  TGNumberEntry     *f_ID;
  TGTextButton      *f_ID_Button;
  TGLabel           *f_ID_Label;

  TGLayoutHints     *fHint1, *fHint2, *fHint3;
  TGListBox         *f_ChannelID;

  TGGroupFrame      *f_Global_Group;
  TGGroupFrame      *f_XYZ_Group;
  TGHorizontalFrame *f_XYZ_X_Group, *f_XYZ_Y_Group, *f_XYZ_Z_Group;
  TGNumberEntry     *f_XYZ_X, *f_XYZ_Y, *f_XYZ_Z;
  TGLabel           *f_XYZ_X_Label, *f_XYZ_Y_Label, *f_XYZ_Z_Label;
  TGTextButton      *f_XYZ_Button;

  Bool_t fClose;

  int _channelID;
  int _iX;
  int _iY;
  
  MusEcalGUI* _gui;

public:

  MEChanPanel( const TGWindow* p, MusEcalGUI* main, UInt_t w, UInt_t h );
  virtual ~MEChanPanel();

  // slots
  void CloseWindow();
  void DoClose();
  void SelectChannel( Int_t );
  void SelectID();
  void SelectXY();
  void SelectXYZ();

ClassDef(MEChanPanel,0) // MEChanPanel -- 

};

#endif
