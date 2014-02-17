// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowTaskExecutor
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jul 11 12:09:47 EDT 2008
// $Id: CmsShowTaskExecutor.cc,v 1.2 2008/11/06 22:05:24 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/src/CmsShowTaskExecutor.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CmsShowTaskExecutor::CmsShowTaskExecutor()
{
}

// CmsShowTaskExecutor::CmsShowTaskExecutor(const CmsShowTaskExecutor& rhs)
// {
//    // do actual copying here;
// }

CmsShowTaskExecutor::~CmsShowTaskExecutor()
{
}

//
// assignment operators
//
// const CmsShowTaskExecutor& CmsShowTaskExecutor::operator=(const CmsShowTaskExecutor& rhs)
// {
//   //An exception safe implementation is
//   CmsShowTaskExecutor temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
CmsShowTaskExecutor::addTask(const TaskFunctor& iTask)
{
   m_tasks.push_back(iTask);
}

void
CmsShowTaskExecutor::startDoingTasks()
{
   if(m_tasks.size()) {
      requestNextTask();
   }
}

void
CmsShowTaskExecutor::doNextTaskImp()
{
   TaskFunctor f = m_tasks.front();
   m_tasks.pop_front();
   f();
}

bool
CmsShowTaskExecutor::moreTasksAvailable()
{
   return !m_tasks.empty();
}

//
// const member functions
//

//
// static member functions
//
