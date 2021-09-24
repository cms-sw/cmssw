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
//

// system include files
#include <deque>
#include <functional>

// user include files
#include "Fireworks/Core/interface/CmsShowTaskExecutorBase.h"

// forward declarations

class CmsShowTaskExecutor : public CmsShowTaskExecutorBase {
public:
  CmsShowTaskExecutor();
  ~CmsShowTaskExecutor() override;

  typedef std::function<void()> TaskFunctor;
  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void addTask(const TaskFunctor& iTask);

  void startDoingTasks() override;

protected:
  void doNextTaskImp() override;
  bool moreTasksAvailable() override;

public:
  CmsShowTaskExecutor(const CmsShowTaskExecutor&) = delete;  // stop default

  const CmsShowTaskExecutor& operator=(const CmsShowTaskExecutor&) = delete;  // stop default

private:
  // ---------- member data --------------------------------
  std::deque<TaskFunctor> m_tasks;
};

#endif
