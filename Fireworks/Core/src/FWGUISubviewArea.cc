// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGUISubviewArea
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Feb 15 14:13:33 EST 2008
// $Id: FWGUISubviewArea.cc,v 1.13 2008/11/05 09:24:25 chrjones Exp $
//

// system include files
#include <assert.h>
#include <stdexcept>
#include <iostream>

#include "TSystem.h"
#include "TGButton.h"
#include "TGSplitFrame.h"
#include "TGFont.h"
#include "TGLabel.h"

// user include files
#include "Fireworks/Core/interface/FWGUISubviewArea.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWGUISubviewArea::FWGUISubviewArea(unsigned int iIndex, const TGSplitFrame *iParent, TGSplitFrame* iMainSplit)
: TGVerticalFrame(iParent),
  m_mainSplit(iMainSplit),
  m_index(iIndex),
  m_docked(true)
{
   //This doesn't seem to do anything
   //SetCleanup(kNoCleanup);

   const unsigned int kIconHeight = 20;
   m_buttons = new TGHorizontalFrame(this);
   this->AddFrame(m_buttons, new TGLayoutHints(kLHintsTop|kLHintsLeft|kLHintsExpandX));
   m_label = new TGLabel(m_buttons,"");
   m_buttons->AddFrame(m_label, new TGLayoutHints(kLHintsLeft|kLHintsTop|kLHintsExpandX|kLHintsExpandY));
   TGPictureButton* temp;
   m_infoButton = temp = new TGPictureButton(m_buttons,infoIcon());
   temp->SetDisabledPicture(infoDisabledIcon());
   m_buttons->AddFrame(m_infoButton, new TGLayoutHints(kLHintsTop|kLHintsLeft|kLHintsExpandY));
   m_infoButton->AllowStayDown(kTRUE);
   m_infoButton->Connect("Pressed()","FWGUISubviewArea",this,"selectButtonDown()");
   m_infoButton->Connect("Released()","FWGUISubviewArea",this,"selectButtonUp()");
   m_infoButton->SetToolTipText("Edit View");

   //have to stop cleanup so that we don't delete the button which was clicked to tell us to delete
   //m_buttons->SetCleanup(kNoCleanup);
   m_swapButton= temp=new TGPictureButton(m_buttons, swapIcon());
   temp->SetDisabledPicture(swapDisabledIcon());

   m_swapButton->SetToolTipText("Swap to big view");
   m_swapButton->SetHeight(kIconHeight);
   m_buttons->AddFrame(m_swapButton, new TGLayoutHints(kLHintsTop|kLHintsLeft|kLHintsExpandY));
   m_swapButton->Connect("Clicked()","FWGUISubviewArea",this,"swapToBigView()");

   m_undockButton = temp = new TGPictureButton(m_buttons,undockIcon());
   temp->SetDisabledPicture(undockDisabledIcon());
   m_undockButton->SetToolTipText("Undock view to own window");
   m_undockButton->SetHeight(kIconHeight);
   m_buttons->AddFrame(m_undockButton, new TGLayoutHints(kLHintsTop|kLHintsLeft|kLHintsExpandY));
   m_undockButton->Connect("Clicked()", "FWGUISubviewArea",this,"undock()");

#if defined(__APPLE__)
   //There is a problem with undocking on OS X
   m_undockButton->SetEnabled(kFALSE);
#endif

   m_closeButton = temp = new TGPictureButton(m_buttons,closeIcon());
   temp->SetDisabledPicture(closeDisabledIcon());
   m_closeButton->SetToolTipText("Close view");
   m_closeButton->SetHeight(kIconHeight);
   m_buttons->AddFrame(m_closeButton, new TGLayoutHints(kLHintsRight|kLHintsTop|kLHintsExpandY));
   m_closeButton->Connect("Clicked()", "FWGUISubviewArea",this,"destroy()");

   //Turn off until we can get this to work consistently correct
   m_closeButton->SetEnabled(kFALSE);
   //behavior of buttons depends on index
   if(0==iIndex) {
      m_swapButton->SetEnabled(kFALSE);
   }
   m_buttons->SetBackgroundColor(TGFrame::GetBlackPixel());
}

// FWGUISubviewArea::FWGUISubviewArea(const FWGUISubviewArea& rhs)
// {
//    // do actual copying here;
// }

FWGUISubviewArea::~FWGUISubviewArea()
{
   //std::cout <<"IN dstr FWGUISubviewArea"<<std::endl;
   m_swapButton->Disconnect("Clicked()",this,"swapToBigView()");
   m_undockButton->Disconnect("Clicked()",this,"undock()");
   m_closeButton->Disconnect("Clicked()", this,"destroy()");
   m_infoButton->Disconnect("Pressed()",this,"selectButtonDown()");
   m_infoButton->Disconnect("Released()",this,"selectButtonUp()");


   //delete m_swapButton;
   //delete m_undockButton;
   //HELP how do I get this to be deleted after we finish processing this GUI event?
   //RemoveFrame(m_closeButton);
   //delete m_closeButton;
   m_closeButton->UnmapWindow();
   m_buttons->RemoveFrame(m_closeButton);
   //std::cout <<"OUT dstr FWGUISubviewArea"<<std::endl;
}

//
// assignment operators
//
// const FWGUISubviewArea& FWGUISubviewArea::operator=(const FWGUISubviewArea& rhs)
// {
//   //An exception safe implementation is
//   FWGUISubviewArea temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWGUISubviewArea::selectButtonDown()
{
      selected_(index());
}

void
FWGUISubviewArea::selectButtonUp()
{
   unselected_(index());
}


void
FWGUISubviewArea::setName(const std::string& iName)
{
   m_label->SetText(iName.c_str());
}

void
FWGUISubviewArea::unselect()
{
   m_infoButton->SetDown(kFALSE);
}


void
FWGUISubviewArea::enableDestructionButton(bool iState)
{
   m_closeButton->SetEnabled(iState);
}

void
FWGUISubviewArea::enableSwapButton(bool iState)
{
   m_swapButton->SetEnabled(iState);
}

void
FWGUISubviewArea::swapToBigView()
{
   //We know the parent is a TGSplitFrame because the constructor requires it to be so
   TGSplitFrame* p = const_cast<TGSplitFrame*>(static_cast<const TGSplitFrame*>(GetParent()));
   p->SwitchToMain();

   swappedToBigView_(index());
}

void
FWGUISubviewArea::destroy()
{
   goingToBeDestroyed_(index());

   //NOTE: FWGUIManager will actually handle the deletion of the window in order to avoid a
   // button sending a signal which causes the button to be destroyed leading to a memory error problem
}

void
FWGUISubviewArea::undock()
{
   //We know the parent is a TGSplitFrame because the constructor requires it to be so
   TGSplitFrame* p = const_cast<TGSplitFrame*>(static_cast<const TGSplitFrame*>(GetParent()));
   m_undockedSwappableView = m_swapButton->IsEnabled();
   m_undockedDestructabledView = m_closeButton->IsEnabled();

   m_swapButton->SetEnabled(kFALSE);
   m_closeButton->SetEnabled(kFALSE);
   m_undockButton->SetEnabled(kFALSE);

   m_docked = false;
   if(index() == 0 ) {
      bigViewUndocked_();
   }
   p->Connect("Docked(TGFrame*)","FWGUISubviewArea",this,"beingDocked(TGFrame*)");
   p->ExtractFrame();

}

void
FWGUISubviewArea::undockTo(Int_t x, Int_t y,
                           UInt_t width, UInt_t height)
{
   undock();
   //NOTE: this seems evil but I can do the exact same thing by calling 'GetId' on the MainFrame
   // and then use gVirtualX to do the work
   const_cast<TGWindow*>(GetMainFrame())->MoveResize(x,y,width,height);
}


void
FWGUISubviewArea::beingDocked(TGFrame*)
{
   m_swapButton->SetEnabled(m_undockedSwappableView);
   m_closeButton->SetEnabled(m_undockedDestructabledView);
   m_undockButton->SetEnabled(kTRUE);
   TGSplitFrame* p = const_cast<TGSplitFrame*>(static_cast<const TGSplitFrame*>(GetParent()));
   p->Disconnect("Docked(TGFrame*)",this,"beingDocked()");

   m_docked=true;
   if(index() == 0 ) {
      bigViewDocked_();
   }
}

//
// const member functions
//
bool
FWGUISubviewArea::isSelected() const
{
   return m_infoButton->IsDown();
}

//
// static member functions
//
 const TGPicture *
FWGUISubviewArea::swapIcon()
{
   static const TGPicture* s_icon = 0;
   if(0== s_icon) {
      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }
      TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));
      //s_icon = gClient->GetPicture(coreIcondir+"swapToMainView.gif");
      s_icon = gClient->GetPicture(coreIcondir+"moveup.png");
   }
   return s_icon;
}

const TGPicture *
FWGUISubviewArea::swapDisabledIcon()
{
   static const TGPicture* s_icon = 0;
   if(0== s_icon) {
      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }
      TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));
      //s_icon = gClient->GetPicture(coreIcondir+"swapToMainView.gif");
      s_icon = gClient->GetPicture(coreIcondir+"moveup-disabled.png");
   }
   return s_icon;
}

const TGPicture *
FWGUISubviewArea::closeIcon()
{
   static const TGPicture* s_icon = 0;
   if(0== s_icon) {
      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }
      TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));
      //s_icon = gClient->GetPicture(coreIcondir+"closeView.gif");
      s_icon = gClient->GetPicture(coreIcondir+"delete.png");
   }
   return s_icon;
}
const TGPicture *
FWGUISubviewArea::closeDisabledIcon()
{
   static const TGPicture* s_icon = 0;
   if(0== s_icon) {
      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }
      TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));
      //s_icon = gClient->GetPicture(coreIcondir+"closeView.gif");
      s_icon = gClient->GetPicture(coreIcondir+"delete-disabled.png");
   }
   return s_icon;
}


const TGPicture *
FWGUISubviewArea::undockIcon()
{
   static const TGPicture* s_icon = 0;
   if(0== s_icon) {
      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }
      TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));
      //s_icon = gClient->GetPicture(coreIcondir+"undockView.gif");
      s_icon = gClient->GetPicture(coreIcondir+"expand.png");
   }
   return s_icon;
}

const TGPicture *
FWGUISubviewArea::undockDisabledIcon()
{
   static const TGPicture* s_icon = 0;
   if(0== s_icon) {
      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }
      TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));
      //s_icon = gClient->GetPicture(coreIcondir+"undockView.gif");
      s_icon = gClient->GetPicture(coreIcondir+"expand-disabled.png");
   }
   return s_icon;
}

const TGPicture *
FWGUISubviewArea::infoIcon()
{
   static const TGPicture* s_icon = 0;
   if(0== s_icon) {
      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }
      TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));
      s_icon = gClient->GetPicture(coreIcondir+"info.png");
   }
   return s_icon;
}

const TGPicture *
FWGUISubviewArea::infoDisabledIcon()
{
   static const TGPicture* s_icon = 0;
   if(0== s_icon) {
      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }
      TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));
      s_icon = gClient->GetPicture(coreIcondir+"info-disabled.png");
   }
   return s_icon;
}


void
FWGUISubviewArea::setIndex(unsigned int iIndex) {
   if(0==iIndex) {
      m_swapButton->SetEnabled(kFALSE);
   }
   if(m_index==0) {
      m_swapButton->SetEnabled(kTRUE);
   }
   m_index = iIndex;
}
