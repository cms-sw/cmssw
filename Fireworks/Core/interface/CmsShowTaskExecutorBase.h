#ifndef Fireworks_Core_CmsShowTaskExecutorBase_h
#define Fireworks_Core_CmsShowTaskExecutorBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowTaskExecutorBase
//
/**\class CmsShowTaskExecutorBase CmsShowTaskExecutorBase.h Fireworks/Core/interface/CmsShowTaskExecutorBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Jul 11 12:09:38 EDT 2008
//

// system include files
#include <sigc++/signal.h>

// user include files

// forward declarations
class TTimer;
class CmsShowTaskTimer;

class CmsShowTaskExecutorBase {
public:
  CmsShowTaskExecutorBase();
  virtual ~CmsShowTaskExecutorBase();

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void requestNextTask();
  void doNextTask();

  virtual void startDoingTasks() = 0;

  sigc::signal<void()> tasksCompleted_;

protected:
  virtual void doNextTaskImp() = 0;
  virtual bool moreTasksAvailable() = 0;

public:
  CmsShowTaskExecutorBase(const CmsShowTaskExecutorBase&) = delete;  // stop default

  const CmsShowTaskExecutorBase& operator=(const CmsShowTaskExecutorBase&) = delete;  // stop default
private:
  // ---------- member data --------------------------------
  //TTimer* m_timer;
  CmsShowTaskTimer* m_taskTimer;
};

#endif
