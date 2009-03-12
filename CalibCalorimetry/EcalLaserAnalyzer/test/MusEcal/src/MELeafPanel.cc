#include <iostream>
using namespace std;

#include "MELeafPanel.hh"
#include "../../interface/MEGeom.h"

#include "MusEcalGUI.hh"
#include <TSystem.h>

ClassImp(MELeafPanel)

MELeafPanel::MELeafPanel( const TGWindow *p, MusEcalGUI* main,
			  UInt_t w, UInt_t h)
:  _gui( main )
{
  _type  = _gui->_type;
  _color = _gui->_color;
  _var   = _gui->_var;
  _zoom  = _gui->_zoom;

  fMain=0;
  fVframe1=0;
  fHframe1=0;
  fHframe2=0;
  fVarBox=0;
  fZoomBox=0;
  fPlotButton=0;
  fOneLevelUpButton=0;

  fDiffPlotButton=0;

  fHint1 = new TGLayoutHints( kLHintsTop | kLHintsCenterX, 5, 5, 5, 5 );
  fHint3 = new TGLayoutHints( kLHintsTop | kLHintsLeft , 3 ,3 ,3 ,3 ); // buttons in groups
  fHint2 = new TGLayoutHints( kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 3, 0);
  fHint4 = new TGLayoutHints( kLHintsTop | kLHintsLeft | kLHintsExpandX,
			      2, 2, 0, 0);
  fHint5 =  new TGLayoutHints( kLHintsCenterX, 0, 0, 0, 0 );

  fMain = new TGTransientFrame(p, main, w, h);
  fMain->Connect( "CloseWindow()", "MELeafPanel", this, "DoClose()" );
  fMain->DontCallClose();
  fMain->SetCleanup(kDeepCleanup);
  
  fMain->ChangeOptions( (fMain->GetOptions() & ~kVerticalFrame) | kHorizontalFrame );
  
  fVframe1 = new TGVerticalFrame( fMain, 0, 0, 0 );
  fHframe1 = new TGHorizontalFrame( fVframe1, 0, 0, 0 );
  fHframe2 = new TGHorizontalFrame( fVframe1, 0, 0, 0 );
  
  fVarBox = new TGListBox( fHframe1, 150, 100 );
  fVarBox->Connect( "Selected(Int_t)", "MELeafPanel", this, "SetVar(UInt_t)" );

  fZoomBox = new TGListBox( fHframe1, 150, 100 );
  fZoomBox->Connect( "Selected(Int_t)", "MELeafPanel", this, "SetZoom(UInt_t)" );
  
  if( _type==ME::iLaser )
    {
      _var =  MusEcal::iAPD;
      for( int jj=0; jj<MusEcal::iSizeLV; jj++ )
	{
	  fVarBox->AddEntry( MusEcal::historyVarName[jj], jj );
	}  
      fVarBox->Resize(150,100);
      fVarBox->Select( _var );
      fVarBox->Layout();

      _zoom = MusEcal::historyVarZoom[_color][ _var ];
      if( _zoom==MusEcal::iZero ) _zoom = MusEcal::iThirtyPercent;
      for( int iZoom=0; iZoom<MusEcal::iZero; iZoom++ )
	{
	  fZoomBox->AddEntry( MusEcal::zoomName[iZoom], iZoom );
	}
      fZoomBox->Resize(150,100);
      fZoomBox->Select( _zoom );
      fZoomBox->Layout();
    }
  else if( _type==ME::iTestPulse )
    {
      int jj_Var =  0;
      for( int jj=0; jj<MusEcal::iSizeTPV; jj++ )
	{
	  fVarBox->AddEntry( MusEcal::historyTPVarName[jj], jj );
	}  
      fVarBox->Resize(150,100);
      fVarBox->Select( jj_Var );
      fVarBox->Layout();

      int jj_Zoom = MusEcal::historyTPVarZoom[ jj_Var ];
      for( int iZoom=0; iZoom<MusEcal::iZero; iZoom++ )
	{
	  fZoomBox->AddEntry( MusEcal::zoomName[iZoom], iZoom );
	}
      fZoomBox->Resize(150,100);
      if( jj_Zoom!=MusEcal::iZero ) fZoomBox->Select( jj_Zoom );
      else fZoomBox->Select(0);
      fZoomBox->Layout();
    }

  fPlotButton = new TGTextButton( fHframe2      , "   Plot      " );
  fPlotButton->Connect("Clicked()","MELeafPanel",this,"DoPlot()");

  //  fDiffPlotButton = new TGTextButton( fHframe2      , "   Diff Plot      " );
  //  fDiffPlotButton->Connect("Clicked()","MELeafPanel",this,"DoDiffPlot()");

  fOneLevelUpButton = new TGTextButton( fHframe2, "  One Level Up  " );
  fOneLevelUpButton->Connect("Clicked()","MELeafPanel",this,"DoOneLevelUp()");

  fHframe1->AddFrame( fVarBox,  fHint3 );
  fHframe1->AddFrame( fZoomBox, fHint3 );
  fHframe1->Resize(310,110);
  fHframe1->MapSubwindows();  
  fVframe1->AddFrame( fHframe1, fHint5 );
  fHframe2->AddFrame( fPlotButton,  fHint3 );
  //  fHframe2->AddFrame( fDiffPlotButton,  fHint3 );
  fHframe2->AddFrame( fOneLevelUpButton, fHint3 );
  fHframe2->Resize(310,50);
  fHframe2->MapSubwindows();  
  fVframe1->AddFrame( fHframe2, fHint5 );
  fVframe1->MapSubwindows();  
  fMain->AddFrame( fVframe1,    fHint5 );

  TString str;
  str += ME::type[_type];
  str += "  Zoom Panel";
  fMain->SetWindowName(str);
  TGDimension size = fMain->GetDefaultSize();
  fMain->Resize(size);

  fMain->MapSubwindows();
  fMain->MapWindow();

}

MELeafPanel::~MELeafPanel()
{
  _gui->_fLeafPanel=0;
  fMain->DeleteWindow();   // deletes fMain
}

void 
MELeafPanel::CloseWindow()
{
  // Called when window is closed via the window manager.

  delete this;
}

void 
MELeafPanel::DoClose()
{
  CloseWindow();
}

void
MELeafPanel::DoPlot()
{
  // cout << "Entering DoPlot for var=" << _var << " and zoom=" << _zoom << endl;
  _gui->_var  = _var;
  _gui->_zoom = _zoom;
  _gui->leafPlot(1);
}

void
MELeafPanel::DoDiffPlot()
{
  // cout << "Plot intervals for type=" << _type << " and var=" << MusEcal::historyVarName[_var] << endl;
  // _gui->_type    = _type;
  //  _gui->setMtqVar(_var);
  //  _gui->_zoom    = _zoom;
  //  cout << "Obsolete -- fixme!!! " << endl;
  //  _gui->intervalPlot();
  //  _gui->distancePlot();
  //  _gui->correctionPlot();
}

void
MELeafPanel::DoOneLevelUp()
{
  _gui->oneLevelUp();
  DoPlot();
}

void
MELeafPanel::SetVar( int ii )
{
  _var = ii;
}

void
MELeafPanel::SetZoom( int iZoom )
{
  _zoom = iZoom;
}
