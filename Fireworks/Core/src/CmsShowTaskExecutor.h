#ifndef Fireworks_Core_CmsShowTaskExecutor_h
#define Fireworks_Core_CmsShowTaskExecutor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowTaskExecutor
//
/**\class CmsShowTaskExecutor CmsShowTaskExecutor.h Fireworks/Core/interface/CmsShowTaskExecutor.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Jul 11 12:09:45 EDT 2008
// $Id: CmsShowTaskExecutor.h,v 1.3 2009/01/23 21:35:42 amraktad Exp $
//

// system include files
#include <boost/function.hpp>
#include <deque>

// user include files
#include "Fireworks/Core/src/CmsShowTaskExecutorBase.h"

// forward declarations

class CmsShowTaskExecutor : public CmsShowTaskExecutorBase {

public:
   CmsShowTaskExecutor();
   virtual ~CmsShowTaskExecutor();

   typedef boost::function0<void> TaskFunctor;
   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void addTask(const TaskFunctor& iTask);

   void startDoingTasks();
protected:
   virtual void doNextTaskImp();
   virtual bool moreTasksAvailable();

private:
   CmsShowTaskExecutor(const CmsShowTaskExecutor&); // stop default

   const CmsShowTaskExecutor& operator=(const CmsShowTaskExecutor&); // stop default

   // ---------- member data --------------------------------
   std::deque<TaskFunctor> m_tasks;
};


#endif
