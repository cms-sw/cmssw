#include <iostream>
using namespace std;

#include "MEMultiVarPanel.hh"

#include "MusEcalGUI.hh"
#include <TSystem.h>

ClassImp(MEMultiVarPanel)

  MEMultiVarPanel::MEMultiVarPanel(const TGWindow *p, MusEcalGUI* main,
		       UInt_t w, UInt_t h)
    :  _gui( main )
{
  _type = _gui->_type;
  _color = _gui->_color;

  fMain=0;
  fVframe1=0;

  fHint1 = new TGLayoutHints( kLHintsTop | kLHintsCenterX, 5, 5, 5, 5 );
  fHint3 = new TGLayoutHints( kLHintsTop | kLHintsLeft , 0 ,0 ,0 ,0 ); // buttons in groups
  fHint2 = new TGLayoutHints( kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0);
  fHint4 = new TGLayoutHints( kLHintsTop | kLHintsLeft | kLHintsExpandX,
			      0, 0, 0, 0);
  fHint5 =  new TGLayoutHints( kLHintsCenterX, 0, 0, 0, 0 );


  fClose = kTRUE;

  fMain = new TGTransientFrame(p, main, w, h);
  fMain->Connect( "CloseWindow()", "MEMultiVarPanel", this, "DoClose()" );
  fMain->DontCallClose();
  fMain->SetCleanup(kDeepCleanup);

  fMain->ChangeOptions( (fMain->GetOptions() & ~kVerticalFrame) | kHorizontalFrame );

  fHframe1 = new TGHorizontalFrame( fMain, 0, 0, 0 );
  fMain->AddFrame( fHframe1, fHint5 );
  fVframe1 = new TGVerticalFrame( fHframe1, 0, 0, 0 );
  fHframe1->AddFrame( fVframe1, fHint5 );
  fVframe2 = new TGVerticalFrame( fHframe1, 0, 0, 0 );
  fHframe1->AddFrame( fVframe2, fHint5 );

  TGVerticalFrame* fVframe = fVframe1;

  if( _type==ME::iLaser )
    {
      f_GroupFrame.resize( MusEcal::iSizeLV, 0 );
      f_ComboBox.resize( MusEcal::iSizeLV, 0 );
      for( int jj=0; jj<MusEcal::iSizeLV; jj++ )
	{
	  if( jj>8 ) fVframe = fVframe2;
	  f_GroupFrame[jj] = new TGGroupFrame( fVframe, "", kHorizontalFrame | kRaisedFrame );
	  f_GroupFrame[jj]->SetTitle( MusEcal::historyVarName[jj] );
	  fVframe->AddFrame( f_GroupFrame[jj] );
	  f_ComboBox[jj] = new TGComboBox( f_GroupFrame[jj] , 1000+jj );
	  f_GroupFrame[jj]->AddFrame( f_ComboBox[jj], fHint4 );
      
	  int jj_Zoom = MusEcal::historyVarZoom[_type][ jj ];
	  f_ComboBox[jj]->AddEntry( "- not selected -", MusEcal::iZero );
	  for( int iZoom=0; iZoom<MusEcal::iZero; iZoom++ )
	    {
	      f_ComboBox[jj]->AddEntry( MusEcal::zoomName[iZoom], iZoom );
	      if( iZoom==jj_Zoom ) f_ComboBox[jj]->Select( iZoom );
	    }
	  if( jj_Zoom==MusEcal::iZero ) f_ComboBox[jj]->Select(MusEcal::iZero);
	  f_ComboBox[jj]->Resize(120,20);

	}
    }
  else if( _type==ME::iTestPulse )
    {
      f_GroupFrame.resize( MusEcal::iSizeTPV, 0 );
      f_ComboBox.resize( MusEcal::iSizeTPV, 0 );
      for( int jj=0; jj<MusEcal::iSizeTPV; jj++ )
	{
	  f_GroupFrame[jj] = new TGGroupFrame( fVframe, "", kHorizontalFrame | kRaisedFrame );
	  f_GroupFrame[jj]->SetTitle( MusEcal::historyTPVarName[jj] );
	  fVframe->AddFrame( f_GroupFrame[jj] );
	  f_ComboBox[jj] = new TGComboBox( f_GroupFrame[jj] , 1000+jj );
	  f_GroupFrame[jj]->AddFrame( f_ComboBox[jj], fHint4 );
      
	  int jj_Zoom = MusEcal::historyTPVarZoom[ jj ];
	  f_ComboBox[jj]->AddEntry( "- not selected -", MusEcal::iZero );
	  for( int iZoom=0; iZoom<MusEcal::iZero; iZoom++ )
	    {
	      f_ComboBox[jj]->AddEntry( MusEcal::zoomName[iZoom], iZoom );
	      if( iZoom==jj_Zoom ) f_ComboBox[jj]->Select( iZoom );
	    }
	  if( jj_Zoom==MusEcal::iZero ) f_ComboBox[jj]->Select(MusEcal::iZero);
	  f_ComboBox[jj]->Resize(120,20);
	}
    }

  f_Go_Button = new TGTextButton( fVframe, " Plot " );
  f_Go_Button->Connect("Clicked()","MEMultiVarPanel",this,"DoGo()");
  fVframe->AddFrame( f_Go_Button, fHint3 );

  TString str;
  str += ME::type[_type];
  str += "Multi-Variable Panel";
  fMain->SetWindowName(str);
  TGDimension size = fMain->GetDefaultSize();
  fMain->Resize(size);

  fMain->MapSubwindows();
  fMain->MapWindow();

}

MEMultiVarPanel::~MEMultiVarPanel()
{
  _gui->_fMultiVarPanel=0;
  fMain->DeleteWindow();   // deletes fMain
}

void 
MEMultiVarPanel::CloseWindow()
{
  // Called when window is closed via the window manager.
  delete this;
}

void 
MEMultiVarPanel::DoClose()
{
  CloseWindow();
}

void
MEMultiVarPanel::DoGo()
{
  if( _type==ME::iLaser )
    {
      for( int jj=0; jj<MusEcal::iSizeLV; jj++ )
	{
	  int iZoom = f_ComboBox[jj]->GetSelected();
	  if( iZoom==MusEcal::iZero ) 
	    {
	      MusEcal::historyVarZoom[_color][jj] = MusEcal::iZero;
	      continue;
	    }
	  else
	    {
	      MusEcal::historyVarZoom[_color][jj] = iZoom;
	    }
	}
    }
  else if( _type==ME::iTestPulse )
    {
      for( int jj=0; jj<MusEcal::iSizeTPV; jj++ )
	{
	  int iZoom = f_ComboBox[jj]->GetSelected();
	  if( iZoom==MusEcal::iZero ) 
	    {
	      MusEcal::historyTPVarZoom[jj] = MusEcal::iZero;
	      continue;
	    }
	  else
	    {
	      MusEcal::historyTPVarZoom[jj] = iZoom;
	    }
	}
    } 
  _gui->multiVarPlot(1);
}
