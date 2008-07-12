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
// $Id$
//

// system include files
#include <iostream>

// user include files
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
CmsShowTaskExecutorBase::CmsShowTaskExecutorBase()
{
   Connect("requestNextTask()","CmsShowTaskExecutorBase",this,"doNextTask()");
}

// CmsShowTaskExecutorBase::CmsShowTaskExecutorBase(const CmsShowTaskExecutorBase& rhs)
// {
//    // do actual copying here;
// }

CmsShowTaskExecutorBase::~CmsShowTaskExecutorBase()
{
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
   std::cout <<"requestNextTask"<<std::endl;
   Emit("requestNextTask()");
}

void 
CmsShowTaskExecutorBase::doNextTask()
{
   doNextTaskImp();
   if(moreTasksAvailable()) {
      requestNextTask();
   }
}

//
// const member functions
//

//
// static member functions
//
