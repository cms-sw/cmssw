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
// $Id: FWGUISubviewArea.cc,v 1.5 2008/06/25 22:05:05 chrjones Exp $
//

// system include files
#include "TSystem.h"
#include "TGButton.h"
#include "TGSplitFrame.h"
#include <assert.h>
#include <stdexcept>
#include <iostream>

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
  m_index(iIndex)
{
   //This doesn't seem to do anything
   //SetCleanup(kNoCleanup);
   
   const unsigned int kIconHeight = 14;
   m_buttons = new TGHorizontalFrame(this);
   this->AddFrame(m_buttons, new TGLayoutHints(kLHintsTop|kLHintsLeft|kLHintsExpandX));
   //have to stop cleanup so that we don't delete the button which was clicked to tell us to delete
   //m_buttons->SetCleanup(kNoCleanup);
   m_swapButton= new TGPictureButton(m_buttons, swapIcon());
   m_swapButton->SetToolTipText("Swap to big view");
   m_swapButton->SetHeight(kIconHeight);
   m_buttons->AddFrame(m_swapButton, new TGLayoutHints(kLHintsTop|kLHintsLeft));
   m_swapButton->Connect("Clicked()","FWGUISubviewArea",this,"swapToBigView()");

   m_undockButton = new TGPictureButton(m_buttons,undockIcon());
   m_undockButton->SetToolTipText("Undock view to own window");
   m_undockButton->SetHeight(kIconHeight);
   m_buttons->AddFrame(m_undockButton, new TGLayoutHints(kLHintsTop|kLHintsLeft));
   m_undockButton->Connect("Clicked()", "FWGUISubviewArea",this,"undock()");
   
   m_closeButton = new TGPictureButton(m_buttons,closeIcon());
   m_closeButton->SetToolTipText("Close view");
   m_closeButton->SetHeight(kIconHeight);
   m_buttons->AddFrame(m_closeButton, new TGLayoutHints(kLHintsRight|kLHintsTop));
   m_closeButton->Connect("Clicked()", "FWGUISubviewArea",this,"destroy()");
   
   //Turn off until we can get this to work consistently correct
   m_closeButton->SetEnabled(kFALSE);
   //behavior of buttons depends on index
   if(0==iIndex) {
      m_swapButton->SetEnabled(kFALSE);
   }
}

// FWGUISubviewArea::FWGUISubviewArea(const FWGUISubviewArea& rhs)
// {
//    // do actual copying here;
// }

FWGUISubviewArea::~FWGUISubviewArea()
{
   std::cout <<"IN dstr FWGUISubviewArea"<<std::endl;
   m_swapButton->Disconnect("Clicked()",this,"swapToBigView()");
   m_undockButton->Disconnect("Clicked()",this,"undock()");
   m_closeButton->Disconnect("Clicked()", this,"destroy()");

   
   //delete m_swapButton;
   //delete m_undockButton;
   //HELP how do I get this to be deleted after we finish processing this GUI event?
   //RemoveFrame(m_closeButton);
   //delete m_closeButton;
   m_closeButton->UnmapWindow();
   m_buttons->RemoveFrame(m_closeButton);
   std::cout <<"OUT dstr FWGUISubviewArea"<<std::endl;
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
FWGUISubviewArea::enableDestructionButton(bool iState)
{
   m_closeButton->SetEnabled(iState);
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

   //We know the parent is a TGSplitFrame because the constructor requires it to be so
   TGSplitFrame* p = const_cast<TGSplitFrame*>(static_cast<const TGSplitFrame*>(GetParent()));
   p->CloseAndCollapse();
   //the above causes the FWGUISubviewArea to be deleted
   //delete this;
}

void
FWGUISubviewArea::undock()
{
   //We know the parent is a TGSplitFrame because the constructor requires it to be so
   TGSplitFrame* p = const_cast<TGSplitFrame*>(static_cast<const TGSplitFrame*>(GetParent()));
   p->ExtractFrame();
   
}   
//
// const member functions
//

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
      s_icon = gClient->GetPicture(coreIcondir+"swapToMainView.png");
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
      s_icon = gClient->GetPicture(coreIcondir+"closeView.png");
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
      s_icon = gClient->GetPicture(coreIcondir+"undockView.png");
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
