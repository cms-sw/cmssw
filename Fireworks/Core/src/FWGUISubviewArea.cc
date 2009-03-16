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
// $Id: FWGUISubviewArea.cc,v 1.18 2009/03/13 22:41:38 amraktad Exp $
//

// system include files
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <boost/bind.hpp>

#include "TSystem.h"
#include "TGButton.h"
#include "TGSplitFrame.h"
#include "TGFont.h"
#include "TGLabel.h"
#include "TEveWindow.h"

#include "Fireworks/Core/interface/FWGUISubviewArea.h"
#include "Fireworks/Core/interface/FWViewBase.h"
#include  "Fireworks/Core/interface/FWGUIManager.h"

//
// constructors and destructor
//
FWGUISubviewArea::FWGUISubviewArea(TEveCompositeFrame* ef, TGCompositeFrame* parent, Int_t height) :
   TGHorizontalFrame(parent, 20, height),
   m_frame (ef)
{
   UInt_t lh = kLHintsNormal | kLHintsExpandX | kLHintsExpandY;

   // info
   m_infoButton = new TGPictureButton(this,infoIcon());
   m_infoButton->ChangeOptions(kRaisedFrame);
   m_infoButton->SetDisabledPicture(infoDisabledIcon());
   AddFrame(m_infoButton, new TGLayoutHints(lh));
   m_infoButton->AllowStayDown(kTRUE);
   m_infoButton->Connect("Pressed()","FWGUISubviewArea",this,"selectButtonDown()");
   m_infoButton->Connect("Released()","FWGUISubviewArea",this,"selectButtonUp()");
   m_infoButton->SetToolTipText("Edit View");

   //swap
   m_swapButton = new TGPictureButton(this, swapIcon());
   m_swapButton->SetDisabledPicture(swapDisabledIcon());
   m_swapButton->SetToolTipText("Swap with current. Current is selected with left mouse click on frame tollbar.");
   m_swapButton->ChangeOptions(kRaisedFrame);
   m_swapButton->SetHeight(height);
   AddFrame(m_swapButton, new TGLayoutHints(lh));
   m_swapButton->Connect("Clicked()","FWGUISubviewArea",this,"swapWithCurrentView()");

   // undock
   m_undockButton = new TGPictureButton(this, undockIcon());
   m_undockButton->ChangeOptions(kRaisedFrame);
   m_undockButton->SetDisabledPicture(undockDisabledIcon());
   m_undockButton->SetToolTipText("Undock view to own window");
   m_undockButton->SetHeight(height);
   AddFrame(m_undockButton, new TGLayoutHints(lh));
   m_undockButton->Connect("Clicked()", "FWGUISubviewArea",this,"undock()");

   // destroy
   m_closeButton = new TGPictureButton(this,closeIcon());
   m_closeButton->ChangeOptions(kRaisedFrame);
   m_closeButton->SetToolTipText("Close view");
   m_closeButton->SetHeight(height);
   AddFrame(m_closeButton, new TGLayoutHints(lh));
   m_closeButton->Connect("Clicked()", "FWGUISubviewArea",this,"destroy()");

   
   // gui manager callbacks
   FWGUIManager* mng = FWGUIManager::getGUIManager();
   goingToBeDestroyed_.connect(boost::bind(&FWGUIManager::subviewIsBeingDestroyed,mng,_1));
   selected_.connect(boost::bind(&FWGUIManager::subviewSelected,mng,_1));
   unselected_.connect(boost::bind(&FWGUIManager::subviewUnselected,mng,_1));
   swapWithCurrentView_.connect(boost::bind(&FWGUIManager::subviewSwapWithCurrent,mng,_1));
}

FWGUISubviewArea::~FWGUISubviewArea()
{
   //std::cout <<"IN dstr FWGUISubviewArea"<<std::endl;
   m_closeButton->Disconnect("Clicked()", this,"destroy()");
   m_infoButton->Disconnect("Pressed()",this,"selectButtonDown()");
   m_infoButton->Disconnect("Released()",this,"selectButtonUp()");
}

//______________________________________________________________________________
//
// actions
//

void
FWGUISubviewArea::selectButtonDown()
{
   selected_(this);
}

void
FWGUISubviewArea::selectButtonUp()
{
   unselected_(this);
}

void
FWGUISubviewArea::unselect()
{
   m_infoButton->SetDown(kFALSE);
}

void
FWGUISubviewArea::swapWithCurrentView()
{
   swapWithCurrentView_(this);
}

void
FWGUISubviewArea::destroy()
{
   goingToBeDestroyed_(this);
}

void
FWGUISubviewArea::undock()
{
   TTimer::SingleShot(50, m_frame->GetEveWindow()->ClassName(), m_frame->GetEveWindow(), "UndockWindowDestroySlot()");
}

void
FWGUISubviewArea::undockTo(Int_t x, Int_t y,
                           UInt_t width, UInt_t height)
{
   //NOTE: this seems evil but I can do the exact same thing by calling 'GetId' on the MainFrame
   // and then use gVirtualX to do the work
   undock();
   const_cast<TGWindow*>(GetMainFrame())->MoveResize(x,y,width,height);
}

//
// const member functions
//
bool
FWGUISubviewArea::isSelected() const
{
   return m_infoButton->IsDown();
}


//______________________________________________________________________________

TEveWindow*
FWGUISubviewArea::getEveWindow()
{
   return m_frame->GetEveWindow();
}


FWViewBase*
FWGUISubviewArea::getFWView()
{
   FWViewBase* v = (FWViewBase*)(getEveWindow()->GetUserData());
   if (v)
   {
      //  printf("get view %s \n", v->typeName().c_str());
   }
   return v;
}

//______________________________________________________________________________
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
