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
// $Id$
//

// system include files

// user include files
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

#include "FWCore/Utilities/interface/Likely.h"

using namespace edm;

//
// member functions
//
SerialTaskQueue::~SerialTaskQueue() {
  //be certain all tasks have completed
  bool isEmpty = m_tasks.empty();
  bool isTaskChosen = m_taskChosen;
  if ((not isEmpty and not isPaused()) or isTaskChosen) {
    pushAndWait([]() { return; });
  }
}

bool SerialTaskQueue::resume() {
  if (0 == --m_pauseCount) {
    tbb::task* t = pickNextTask();
    if (nullptr != t) {
      tbb::task::spawn(*t);
    }
    return true;
  }
  return false;
}

void SerialTaskQueue::pushTask(TaskBase* iTask) {
  tbb::task* t = pushAndGetNextTask(iTask);
  if (nullptr != t) {
    tbb::task::spawn(*t);
  }
}

tbb::task* SerialTaskQueue::pushAndGetNextTask(TaskBase* iTask) {
  tbb::task* returnValue{nullptr};
  if
    LIKELY(nullptr != iTask) {
      m_tasks.push(iTask);
      returnValue = pickNextTask();
    }
  return returnValue;
}

tbb::task* SerialTaskQueue::finishedTask() {
  m_taskChosen.store(false);
  return pickNextTask();
}

SerialTaskQueue::TaskBase* SerialTaskQueue::pickNextTask() {
  bool expect = false;
  if
    LIKELY(0 == m_pauseCount and m_taskChosen.compare_exchange_strong(expect, true)) {
      TaskBase* t = nullptr;
      if
        LIKELY(m_tasks.try_pop(t)) { return t; }
      //no task was actually pulled
      m_taskChosen.store(false);

      //was a new entry added after we called 'try_pop' but before we did the clear?
      expect = false;
      if (not m_tasks.empty() and m_taskChosen.compare_exchange_strong(expect, true)) {
        t = nullptr;
        if (m_tasks.try_pop(t)) {
          return t;
        }
        //no task was still pulled since a different thread beat us to it
        m_taskChosen.store(false);
      }
    }
  return nullptr;
}

void SerialTaskQueue::pushAndWait(tbb::empty_task* iWait, TaskBase* iTask) {
  auto nextTask = pushAndGetNextTask(iTask);
  if
    LIKELY(nullptr != nextTask) {
      if
        LIKELY(nextTask == iTask) {
          //spawn and wait for all requires the task to have its parent set
          iWait->spawn_and_wait_for_all(*nextTask);
        }
      else {
        tbb::task::spawn(*nextTask);
        iWait->wait_for_all();
      }
    }
  else {
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
