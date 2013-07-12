// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowTaskTimer
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Joshua Berger
//         Created:  Fri Jul 25 11:49:12 EDT 2008
// $Id: CmsShowTaskTimer.cc,v 1.3 2009/01/23 21:35:42 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/src/CmsShowTaskTimer.h"
#include "Fireworks/Core/src/CmsShowTaskExecutorBase.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CmsShowTaskTimer::CmsShowTaskTimer(CmsShowTaskExecutorBase* taskExec, Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode), m_taskExec(taskExec)
{
}

// CmsShowTaskTimer::CmsShowTaskTimer(const CmsShowTaskTimer& rhs)
// {
//    // do actual copying here;
// }

CmsShowTaskTimer::~CmsShowTaskTimer()
{
}

//
// assignment operators
//
// const CmsShowTaskTimer& CmsShowTaskTimer::operator=(const CmsShowTaskTimer& rhs)
// {
//   //An exception safe implementation is
//   CmsShowTaskTimer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
Bool_t
CmsShowTaskTimer::Notify()
{
   m_taskExec->doNextTask();
   return kTRUE;
}

//
// const member functions
//

//
// static member functions
//
