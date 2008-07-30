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
// $Id$
//

// system include files
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/CSGContinuousAction.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CSGContinuousAction::CSGContinuousAction(CmsShowMainFrame *iFrame, const char *iName):
CSGAction(iFrame,iName),
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
CSGContinuousAction::switchMode()
{
   if(!m_isRunning) {
      m_isRunning = true;
      CSGAction::globalEnable();
      if(getToolBar() && m_runningImageFileName.size()) {
         getToolBar()->ChangeIcon(getToolBarData(),m_runningImageFileName.c_str());
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
