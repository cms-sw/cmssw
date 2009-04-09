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
// $Id: FWGUISubviewArea.cc,v 1.24 2009/04/09 15:09:58 amraktad Exp $
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
   m_frame (ef),
   m_swapButton(0),
   m_undockButton(0), m_dockButton(0),
   m_closeButton(0),
   m_infoButton(0)
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
   m_swapButton->SetToolTipText("Swap. Select window with click on frame toolbar");
   m_swapButton->ChangeOptions(kRaisedFrame);
   m_swapButton->SetHeight(height);
   AddFrame(m_swapButton, new TGLayoutHints(lh));
   m_swapButton->Connect("Clicked()","FWGUISubviewArea",this,"swap()");

   // dock 
   if (dynamic_cast<const TGMainFrame*>(ef->GetParent()))
   {  
      m_dockButton = new TGPictureButton(this, dockIcon());
      m_dockButton->ChangeOptions(kRaisedFrame);
      m_dockButton->ChangeOptions(kRaisedFrame);
      m_dockButton->SetToolTipText("Dock view");
      m_dockButton->SetHeight(height);
      AddFrame(m_dockButton, new TGLayoutHints(lh));
      m_dockButton->Connect("Clicked()", "FWGUISubviewArea",this,"dock()");
   }
   else
   {
      // undock  
      m_undockButton = new TGPictureButton(this, undockIcon());
      m_undockButton->ChangeOptions(kRaisedFrame);
      m_undockButton->SetDisabledPicture(undockDisabledIcon());
      m_undockButton->SetToolTipText("Undock view to own window");
      m_undockButton->SetHeight(height);
      AddFrame(m_undockButton, new TGLayoutHints(lh));
      m_undockButton->Connect("Clicked()", "FWGUISubviewArea",this,"undock()");
   }
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
   swap_.connect(boost::bind(&FWGUIManager::subviewSwapped,mng,_1));
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
FWGUISubviewArea::currentWindowChanged()
{
  m_swapButton->SetEnabled(!m_frame->GetEveWindow()->IsCurrent());
}

void
FWGUISubviewArea::swap()
{
   swap_(this);
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
FWGUISubviewArea::dock()
{
   TTimer::SingleShot(0, m_frame->ClassName(), m_frame, "MainFrameClosed()");
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
void 
FWGUISubviewArea::configurePrimaryView()
{
   m_closeButton->SetEnabled(kFALSE);
   m_undockButton->SetEnabled(kFALSE);
}

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
FWGUISubviewArea::dockIcon()
{
   static const TGPicture* s_icon = 0;
   if(0== s_icon) {
      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }
      TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));
      s_icon = gClient->GetPicture(coreIcondir+"dock.png");
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
