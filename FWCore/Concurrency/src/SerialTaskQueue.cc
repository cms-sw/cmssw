// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     SerialTaskQueue
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:31:52 CST 2013
// $Id: SerialTaskQueue.cc,v 1.1 2013/02/21 22:14:11 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

#include "FWCore/Utilities/interface/Likely.h"

using namespace edm;

//
// member functions
//
bool
SerialTaskQueue::resume() {
  if(0==--m_pauseCount) {
    tbb::task* t = pickNextTask();
    if(0 != t) {
      tbb::task::spawn(*t);
    }
    return true;
  }
  return false;
}

void
SerialTaskQueue::pushTask(TaskBase* iTask) {
  tbb::task* t = pushAndGetNextTask(iTask);
  if(0!=t) {
    tbb::task::spawn(*t);      
  }
}

tbb::task* 
SerialTaskQueue::pushAndGetNextTask(TaskBase* iTask) {
  tbb::task* returnValue{0};
  if likely(0!=iTask) {
    m_tasks.push(iTask);
    returnValue = pickNextTask();
  }
  return returnValue;
}


tbb::task*
SerialTaskQueue::finishedTask() {
  m_taskChosen.clear();
  return pickNextTask();
}

SerialTaskQueue::TaskBase*
SerialTaskQueue::pickNextTask() {
  
  if likely(0 == m_pauseCount and not m_taskChosen.test_and_set()) {
    TaskBase* t=0;
    if likely(m_tasks.try_pop(t)) {
      return t;
    }
    //no task was actually pulled
    m_taskChosen.clear();
    
    //was a new entry added after we called 'try_pop' but before we did the clear?
    if(not m_tasks.empty() and not m_taskChosen.test_and_set()) {
      TaskBase* t=0;
      if(m_tasks.try_pop(t)) {
        return t;
      }
      //no task was still pulled since a different thread beat us to it
      m_taskChosen.clear();
      
    }
  }
  return 0;
}

void SerialTaskQueue::pushAndWait(tbb::empty_task* iWait, TaskBase* iTask) {
   auto nextTask = pushAndGetNextTask(iTask);
   if likely(nullptr != nextTask) {
     if likely(nextTask == iTask) {
        //spawn and wait for all requires the task to have its parent set
        iWait->spawn_and_wait_for_all(*nextTask);
     } else {
        tbb::task::spawn(*nextTask);
        iWait->wait_for_all();
     }
   } else {
     //a task must already be running in this queue
     iWait->wait_for_all();              
   }
   tbb::task::destroy(*iWait);
}


//
// const member functions
//

//
// static member functions
//
