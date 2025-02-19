// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowTaskExecutorBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jul 11 12:09:41 EDT 2008
// $Id: CmsShowTaskExecutorBase.cc,v 1.6 2009/01/23 21:35:42 amraktad Exp $
//

// system include files
#include <iostream>
#include <TTimer.h>

// user include files
#include "Fireworks/Core/src/CmsShowTaskExecutorBase.h"
#include "Fireworks/Core/src/CmsShowTaskTimer.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CmsShowTaskExecutorBase::CmsShowTaskExecutorBase()
//:m_timer( new TTimer(1) )
{
   //m_timer->Connect("Timeout()","CmsShowTaskExecutorBase",this,"doNextTask()");
   m_taskTimer = new CmsShowTaskTimer(this, 10);
}

// CmsShowTaskExecutorBase::CmsShowTaskExecutorBase(const CmsShowTaskExecutorBase& rhs)
// {
//    // do actual copying here;
// }

CmsShowTaskExecutorBase::~CmsShowTaskExecutorBase()
{
   //delete m_timer;
}

//
// assignment operators
//
// const CmsShowTaskExecutorBase& CmsShowTaskExecutorBase::operator=(const CmsShowTaskExecutorBase& rhs)
// {
//   //An exception safe implementation is
//   CmsShowTaskExecutorBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
CmsShowTaskExecutorBase::requestNextTask()
{
   //NOTE: If I use my own timer then the first time I call Start it works but the second
   //  time causes a segmentation fault

   //Emit("requestNextTask()");
   //m_timer->Start(1,kTRUE);
   //std::cout <<"requestNextTask"<<std::endl;
   m_taskTimer->TurnOn();
}

void
CmsShowTaskExecutorBase::doNextTask()
{
   m_taskTimer->TurnOff();
   doNextTaskImp();
   if(moreTasksAvailable()) {
      requestNextTask();
   }
   else {
      tasksCompleted_.emit();
   }
}

//
// const member functions
//

//
// static member functions
//
