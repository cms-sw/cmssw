#include <iostream>
using namespace std;

#include "MERunPanel.hh"

#include "MusEcalGUI.hh"
#include "MERunManager.hh"
#include "MERun.hh"
#include <TSystem.h>

ClassImp(MERunPanel)

MERunPanel::MERunPanel( const TGWindow *p, MusEcalGUI* main,
		        UInt_t w, UInt_t h )
:  _gui( main )
{
  
  fMain=0;
  fVframe1=0;
  f_Run_Group=0;
  fHint1=0;
  fHint2=0;
  fHint3=0;
  f_RunList=0;
  
  fHint1 = new TGLayoutHints( kLHintsTop | kLHintsCenterX, 5, 5, 5, 5 );
  fHint2 = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0);
  fHint3 = new TGLayoutHints( kLHintsTop | kLHintsLeft , 3 ,3 ,3 ,3 );
  
  // Dialog used to test the different supported progress bars.

  fClose = kTRUE;

  fMain = new TGTransientFrame(p, main, w, h);
  fMain->Connect( "CloseWindow()", "MERunPanel", this, "DoClose()" );
  fMain->DontCallClose();
  fMain->SetCleanup(kDeepCleanup);

  fMain->ChangeOptions( (fMain->GetOptions() & ~kVerticalFrame) | kHorizontalFrame );

  fVframe1 = new TGVerticalFrame(fMain, 0, 0, 0);

  f_Run_Group = new TGGroupFrame( fVframe1,"", kHorizontalFrame | kRaisedFrame );
  TString str = "Run Selection Panel";
  f_Run_Group->SetTitle(str);

  f_RunList = new TGListBox( f_Run_Group, 100, 200 ); // was 100 100
  f_RunList->Connect( "Selected(Int_t)", "MERunPanel", this, "SetCurrentRun(UInt_t)" );

  MERunManager* mgr_ = _gui->_runMgr.begin()->second;
  MusEcal::RunIterator it;
  unsigned int ii=0;
  unsigned int irun=0;
  for( it=mgr_->begin(); it!=mgr_->end(); it++ )
    {
      ME::Time time    = it->first;
      MERun*   aRun    = it->second;
      unsigned int run = aRun->run();
      if( run!=irun )
	{
	  irun= run;
	  ii=0;	  
	}
      ii++;
      TString runstr;
      runstr+=irun;
      runstr+="_";
      runstr+=ii;
      f_RunList->AddEntry(runstr,time);
    }
  f_RunList->Resize(100,200);  // was 100 100 
  f_RunList->Select( mgr_->curKey() );
  f_RunList->MapSubwindows();
  f_RunList->Layout();

  fVframe1->Resize(160, 50);

  f_Run_Group->AddFrame( f_RunList, fHint3 );

  fVframe1->AddFrame( f_Run_Group, fHint1 );

  fMain->AddFrame( fVframe1, fHint3 );

  fMain->SetWindowName("Selection Panel");
  TGDimension size = fMain->GetDefaultSize();
  fMain->Resize(size);


  fMain->MapSubwindows();
  fMain->MapWindow();

}

MERunPanel::~MERunPanel()
{
  _gui->_fRunPanel=0;
  fMain->DeleteWindow();   // deletes fMain
}

void 
MERunPanel::CloseWindow()
{
  // Called when window is closed via the window manager.

  delete this;
}

void 
MERunPanel::DoClose()
{
  CloseWindow();
}

void 
MERunPanel::SetCurrentRun( UInt_t key )
{
  _gui->setTime( key );
}

