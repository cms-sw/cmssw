// -*- C++ -*-
//
// Package:     Core
// Class  :     CSGContinuousAction
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jul 29 10:21:18 EDT 2008
// $Id: CSGContinuousAction.cc,v 1.4 2009/01/23 21:35:42 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>
#include "TGMenu.h"

// user include files
#include "Fireworks/Core/interface/CSGContinuousAction.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CSGContinuousAction::CSGContinuousAction(CmsShowMainFrame *iFrame, const char *iName) :
   CSGAction(iFrame,iName),
   m_upPic(0),
   m_downPic(0),
   m_disabledPic(0),
   m_runningUpPic(0),
   m_runningDownPic(0),
   m_button(0),
   m_isRunning(false)
{
   activated.connect(boost::bind(&CSGContinuousAction::switchMode, this));
}

// CSGContinuousAction::CSGContinuousAction(const CSGContinuousAction& rhs)
// {
//    // do actual copying here;
// }

//CSGContinuousAction::~CSGContinuousAction()
//{
//}

//
// assignment operators
//
// const CSGContinuousAction& CSGContinuousAction::operator=(const CSGContinuousAction& rhs)
// {
//   //An exception safe implementation is
//   CSGContinuousAction temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
CSGContinuousAction::createToolBarEntry(TGToolBar *iToolbar, const char *iImageFileName, const char* iRunningImageFileName)
{
   m_imageFileName=iImageFileName;
   m_runningImageFileName=iRunningImageFileName;
   CSGAction::createToolBarEntry(iToolbar,iImageFileName);
}

void
CSGContinuousAction::createCustomIconsButton(TGCompositeFrame* p,
                                             const TGPicture* upPic,
                                             const TGPicture* downPic,
                                             const TGPicture* disabledPic,
                                             const TGPicture* upRunningPic,
                                             const TGPicture* downRunningPic,
                                             TGLayoutHints* l,
                                             Int_t id,
                                             GContext_t norm,
                                             UInt_t option)
{
   m_upPic=upPic;
   m_downPic=downPic;
   m_disabledPic=disabledPic;
   m_runningUpPic=upRunningPic;
   m_runningDownPic=downRunningPic;
   m_button =
      CSGAction::createCustomIconsButton(p,upPic,downPic,disabledPic,l,id,norm,option);
}

void
CSGContinuousAction::switchMode()
{
   if(!m_isRunning) {
      m_isRunning = true;
      CSGAction::globalEnable();
      if(getToolBar() && m_runningImageFileName.size()) {
         getToolBar()->ChangeIcon(getToolBarData(),m_runningImageFileName.c_str());
      }
      if(0!=m_button) {
         const TGPicture* tUp = m_runningUpPic;
         const TGPicture* tDown = m_runningDownPic;
         m_button->swapIcons(tUp,tDown,m_disabledPic);
      }
      if(0!=getMenu()) {
         getMenu()->CheckEntry(getMenuEntry());
      }
      started_();
   } else {
      stop();
      stopped_();
   }
}

void
CSGContinuousAction::stop()
{
   m_isRunning=false;
   if(getToolBar() && m_imageFileName.size()) {
      getToolBar()->ChangeIcon(getToolBarData(),m_imageFileName.c_str());
   }
   if(0!=m_button) {
      const TGPicture* tUp = m_upPic;
      const TGPicture* tDown = m_downPic;

      m_button->swapIcons(tUp,tDown,m_disabledPic);
   }
   if(0!=getMenu()) {
      getMenu()->UnCheckEntry(getMenuEntry());
   }
   
}


void
CSGContinuousAction::globalEnable()
{
   CSGAction::globalEnable();
}

void
CSGContinuousAction::globalDisable()
{
   if(!m_isRunning) {
      CSGAction::globalDisable();
   }
}

//
// const member functions
//

//
// static member functions
//
