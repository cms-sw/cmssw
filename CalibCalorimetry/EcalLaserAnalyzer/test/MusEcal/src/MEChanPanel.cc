#include <algorithm>
#include <vector>
#include <iostream>
#include <cassert>
using namespace std;

#include "MEChanPanel.hh"
#include "../../interface/MEGeom.h"
#include "../../interface/MEChannel.h"

#include "MusEcalGUI.hh"
#include "MERunManager.hh"
#include <TSystem.h>

ClassImp(MEChanPanel)

MEChanPanel::MEChanPanel(const TGWindow *p, MusEcalGUI* main,
			 UInt_t w, UInt_t h)
:  _gui( main )
{
  
  fMain=0;
  fHframe1=0;
  f_Channel_Group=0;
  f_Channel_ID_Group=0;
  f_Channel_XY_Group=0;
  f_X_Group=0;
  f_Y_Group=0;
  f_X=0;
  f_Y=0;
  f_XY_Button=0;
  f_ID_Group=0;
  f_ID=0;
  f_ID_Button=0;
  //  f_Label_ID=0;
  fHint1=0;
  fHint2=0;
  fHint3=0;

  fHint1 = new TGLayoutHints( kLHintsTop | kLHintsLeft | kLHintsCenterX, 3, 3, 20, 1 );
  fHint2 = new TGLayoutHints( kLHintsTop | kLHintsCenterX | kLHintsExpandY, 3, 3, 5, 3);
  fHint3 = new TGLayoutHints( kLHintsTop | kLHintsLeft , 3 , 3 ,5 ,3 );

  // Dialog used to test the different supported progress bars.

  fClose = kTRUE;

  fMain = new TGTransientFrame(p, main, w, h);
  fMain->Connect( "CloseWindow()", "MEChanPanel", this, "DoClose()" );
  fMain->DontCallClose();
  fMain->SetCleanup(kDeepCleanup);

  fMain->ChangeOptions( (fMain->GetOptions() & ~kVerticalFrame) | kHorizontalFrame );

  fHframe1 = new TGHorizontalFrame(fMain, 0, 0, 0);

  f_Channel_Group = new TGGroupFrame( fHframe1,"", kHorizontalFrame );
  f_Channel_Group->SetTitle(" Local ");

  f_Channel_ID_Group = new TGGroupFrame( f_Channel_Group,"", kVerticalFrame );
  f_Channel_ID_Group->SetTitle(" Crystal Number ");

  //  MEChannel* tree_ = _gui->curMgr()->tree();
  //  assert( tree_!=0 && tree_->m()!=0 );
  //  vector<MEChannel*> vec;
  //  tree_->m()->getListOfChannels( vec );  // at the sector level...
  //  sort( vec.begin(), vec.end() );

  f_Channel_XY_Group = new TGGroupFrame( f_Channel_Group,"", kVerticalFrame );
  f_Channel_XY_Group->SetTitle(" ix and iy ");

  f_X_Group = new TGHorizontalFrame( f_Channel_XY_Group, 5, 1 );
  f_Channel_XY_Group->AddFrame( f_X_Group, fHint3 );
  f_X      = new TGNumberEntry(f_X_Group, 0, 6, 1,
			       TGNumberFormat::kNESInteger, 
			       TGNumberFormat::kNEANonNegative,
			       TGNumberFormat::kNELLimitMinMax , 0, 84 );
  f_X_Group->AddFrame(f_X, fHint3 );
  f_X_Label = new TGLabel( f_X_Group, "   ix" );
  f_X_Group->AddFrame( f_X_Label, fHint3 );

  f_Y_Group = new TGHorizontalFrame( f_Channel_XY_Group, 10, 1 );
  f_Channel_XY_Group->AddFrame( f_Y_Group, fHint3 );
  f_Y      = new TGNumberEntry(f_Y_Group, 0, 6, 1,
			       TGNumberFormat::kNESInteger, 
			       TGNumberFormat::kNEANonNegative,
			       TGNumberFormat::kNELLimitMinMax , 0, 19 );
  f_Y_Group->AddFrame(f_Y, fHint3 );
  f_Y_Label = new TGLabel( f_Y_Group, "iy" );
  f_Y_Group->AddFrame( f_Y_Label, fHint3 );

  //   cout << "Creating the button " << endl;

  f_XY_Button = new TGTextButton( f_Channel_XY_Group, "   Select   " );
  f_Channel_XY_Group->AddFrame( f_XY_Button, fHint1 );
  f_XY_Button->Connect("Clicked()","MEChanPanel",this,"SelectXY()");

  // ++++++++++++++++++++++

  f_ID_Group = new TGHorizontalFrame( f_Channel_ID_Group, 10, 1 );
  f_ID      = new TGNumberEntry(f_ID_Group, 1, 6, 1,
				TGNumberFormat::kNESInteger, 
				TGNumberFormat::kNEAPositive,
				TGNumberFormat::kNELLimitMinMax , 1, 1700 );
  f_ID_Group->AddFrame(f_ID, fHint3 );
  f_ID_Label = new TGLabel( f_ID_Group, "   id" );
  f_ID_Group->AddFrame( f_ID_Label, fHint3 );
  f_Channel_ID_Group->AddFrame( f_ID_Group, fHint3 );
 
  f_ID_Button = new TGTextButton( f_Channel_ID_Group, "   Select   " );
  f_ID_Button->Connect("Clicked()","MEChanPanel",this,"SelectID()");
  f_Channel_ID_Group->AddFrame( f_ID_Button, fHint1 );

//   f_Channel_ID = new TGListBox( f_Channel_ID_Group, 70, 80 );
//   f_Channel_ID->Connect("Selected(Int_t)","MEChanPanel",this,
// 		       "SelectChannel(Int_t)");
//   for( unsigned int ii=0; ii<vec.size(); ii++ )
//     {
//       int id = vec[ii]->id();
//       TString str; str+=id;
//       f_Channel_ID->AddEntry(str,id);
//     }
//   f_Channel_ID->Resize(60,90);
//   f_Channel_ID->Select(1);   
//  f_Channel_ID_Group->AddFrame( f_Channel_ID, fHint3 );

  f_XYZ_Group = new TGGroupFrame( fHframe1,"", kVerticalFrame );
  f_XYZ_Group->SetTitle(" Global ");

  f_XYZ_X_Group = new TGHorizontalFrame( f_XYZ_Group, 10, 1 );
  f_XYZ_Group->AddFrame( f_XYZ_X_Group, fHint3 );
  f_XYZ_X      = new TGNumberEntry(f_XYZ_X_Group, 1, 6, 1,
				   TGNumberFormat::kNESInteger, 
				   TGNumberFormat::kNEAAnyNumber,
				   TGNumberFormat::kNELLimitMinMax,-100,100);
  f_XYZ_X_Group->AddFrame( f_XYZ_X, fHint3 );
  f_XYZ_X_Label = new TGLabel( f_XYZ_X_Group, "   iX (iEta)" );
  f_XYZ_X_Group->AddFrame( f_XYZ_X_Label, fHint3 );

  f_XYZ_Y_Group = new TGHorizontalFrame( f_XYZ_Group, 10, 1 );
  f_XYZ_Group->AddFrame( f_XYZ_Y_Group, fHint3 );
  f_XYZ_Y      = new TGNumberEntry(f_XYZ_Y_Group, 1, 6, 1,
				   TGNumberFormat::kNESInteger, 
				   TGNumberFormat::kNEAPositive,
				   TGNumberFormat::kNELLimitMinMax,1,360);
  f_XYZ_Y_Group->AddFrame( f_XYZ_Y, fHint3 );
  f_XYZ_Y_Label = new TGLabel( f_XYZ_Y_Group, "   iY (iPhi)" );
  f_XYZ_Y_Group->AddFrame( f_XYZ_Y_Label, fHint3 );

  f_XYZ_Z_Group = new TGHorizontalFrame( f_XYZ_Group, 10, 1 );
  f_XYZ_Group->AddFrame( f_XYZ_Z_Group, fHint3 );
  f_XYZ_Z      = new TGNumberEntry(f_XYZ_Z_Group, 0, 6, 1,
				   TGNumberFormat::kNESInteger, 
				   TGNumberFormat::kNEAAnyNumber,
				   TGNumberFormat::kNELLimitMinMax,-1,1);
  f_XYZ_Z_Group->AddFrame( f_XYZ_Z, fHint3 );
  f_XYZ_Z_Label = new TGLabel( f_XYZ_Z_Group, "   iZ" );
  f_XYZ_Z_Group->AddFrame( f_XYZ_Z_Label, fHint3 );

  //   cout << "Creating the button " << endl;

  f_XYZ_Button = new TGTextButton( f_XYZ_Group, "   Select   " );
  f_XYZ_Group->AddFrame( f_XYZ_Button, fHint1 );
  f_XYZ_Button->Connect("Clicked()","MEChanPanel",this,"SelectXYZ()");

  //   f_ChannelXY_Group->Resize(50,30);

  //  fHframe1->Resize(30, 30);

  f_Channel_Group->AddFrame( f_Channel_ID_Group, fHint2 );   
  f_Channel_Group->AddFrame( f_Channel_XY_Group, fHint2 );   

  //  f_Global_Group->AddFrame( f_XYZ_Group, fHint2 );

  //    f_ChannelXY_Group->AddFrame( f_X_Group, fHint2 );
  //    f_ChannelXY_Group->AddFrame( f_Y_Group, fHint2 );

  //   fHframe1->AddFrame( f_RunList,   fHint1 );
  fHframe1->AddFrame( f_Channel_Group, fHint2 );
  fHframe1->AddFrame( f_XYZ_Group,  fHint2 );

  fMain->AddFrame( fHframe1, fHint2 );

  fMain->SetWindowName("Channel Selection Panel");
  TGDimension size = fMain->GetDefaultSize();
  fMain->Resize(size);

  // position relative to the parent's window
  //   fMain->CenterOnParent();

  fMain->MapSubwindows();
  fMain->MapWindow();
}

MEChanPanel::~MEChanPanel()
{
  _gui->_fChanPanel=0;
  fMain->DeleteWindow();   // deletes fMain
}

void 
MEChanPanel::CloseWindow()
{
  // Called when window is closed via the window manager.
  delete this;
}

void 
MEChanPanel::DoClose()
{
  CloseWindow();
}

void 
MEChanPanel::SelectXY()
{
  //  _gui->_iG = MusEcalHist::iChannel;
  int iX = (int)f_X->GetNumber();
  int iY = (int)f_Y->GetNumber();
  cout << "Select iX=" << iX << " and iY=" << iY << endl;
  //  int channel = _gui->setChannelXY( iX, iY );
  // FIXME!!!
  MEChannel* lmr_ = _gui->curMgr()->tree(); // int ilmr = lmr_->id();
  MEChannel* sect_ = lmr_->m(); int isect = sect_->id();
  MEChannel* reg_  = sect_->m(); int ireg = reg_->id();
  int ix = iX;						
  int iy = iY;
  if( ireg==ME::iEBM || ireg==ME::iEBP )
    {
      MEEBGeom::EtaPhiCoord etaphi_ = MEEBGeom::globalCoord( isect, iX, iY );
      ix = etaphi_.first;
      iy = etaphi_.second;      
    }
  MEChannel* chan_ = _gui->curMgr()->tree()->getChannel( ix, iy );
  if( chan_==0 ) 
    {						
      cout << "channel not found" << endl;
      return;
    }
  _gui->setChannel( chan_ );
  //  f_ChannelID->Select( channel );
}

void 
MEChanPanel::SelectXYZ()
{
  int iX = (int)f_XYZ_X->GetNumber();
  int iY = (int)f_XYZ_Y->GetNumber();
  int iZ = (int)f_XYZ_Z->GetNumber();
  cout << "Select iX=" << iX << ", iY=" << iY << " and iZ=" << iZ << endl;
  if( iZ!=0 ) 
    {
      cout << "End-caps not implemented yet" << endl;
      return;
    }
  int iphi=iY;
  int ieta=iX;
  int ireg=0;
  assert( iphi>0 );
  assert( ieta!=0 && std::abs(ieta)<=85 );
  if( ieta>0 )      ireg=ME::iEBP;
  else if( ieta<0 ) ireg=ME::iEBM;
  vector< MEChannel* > vec;
  ME::regTree( ireg )->getListOfChannels( vec );
  MEChannel* chan_(0);
  for( unsigned int ii=0; ii<vec.size(); ii++ )
    {
      MEChannel* leaf_ = vec[ii];
      if( leaf_->ix()!=ieta ) continue;
      if( leaf_->iy()!=iphi ) continue;
      chan_ = leaf_;
    }
  if( chan_==0 ) return;
  _gui->setChannel( chan_ );
}


void 
MEChanPanel::SelectChannel( Int_t channel )
{
  MEChannel* chan_ = _gui->curMgr()->tree()->m()
    ->getDescendant( ME::iCrystal, channel );
  if( chan_==0 ) 
    {						
      cout << "channel not found: " << channel <<  endl;
      return;
    }
  //  int iX=chan_->ix();
  //  int iY=chan_->iy();
  //  f_X->SetNumber( iX );
  //  f_Y->SetNumber( iY );
  //  cout << "Select channel=" << channel << " ---> iX=" << iX << " and iY=" << iY << endl;
  _gui->setChannel( chan_ );
}

void 
MEChanPanel::SelectID()
{
  //  _gui->_iG = MusEcalHist::iChannel;
  int ID = (int)f_ID->GetNumber();
  cout << "Select ID=" << ID << endl;
  SelectChannel( ID );
}
