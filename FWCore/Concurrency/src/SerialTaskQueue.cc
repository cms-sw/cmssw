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
#include "oneapi/tbb/task_group.h"

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
    oneapi::tbb::task_group g;
    tbb::task_handle last{g.defer([]() {})};
    push(g, [&g, &last]() { g.run(std::move(last)); });
    g.wait();
  }
}

void SerialTaskQueue::spawn(TaskBase& iTask) {
  auto pTask = &iTask;
  iTask.group()->run([pTask, this]() {
    TaskBase* t = pTask;
    auto g = pTask->group();
    do {
      t->execute();
      delete t;
      t = finishedTask();
      if (t and t->group() != g) {
        spawn(*t);
        t = nullptr;
      }
    } while (t != nullptr);
  });
}

bool SerialTaskQueue::resume() {
  if (0 == --m_pauseCount) {
    auto t = pickNextTask();
    if (nullptr != t) {
      spawn(*t);
    }
    return true;
  }
  return false;
}

void SerialTaskQueue::pushTask(TaskBase* iTask) {
  auto t = pushAndGetNextTask(iTask);
  if (nullptr != t) {
    spawn(*t);
  }
}

SerialTaskQueue::TaskBase* SerialTaskQueue::pushAndGetNextTask(TaskBase* iTask) {
  TaskBase* returnValue{nullptr};
  if LIKELY (nullptr != iTask) {
    m_tasks.push(iTask);
    returnValue = pickNextTask();
  }
  return returnValue;
}

SerialTaskQueue::TaskBase* SerialTaskQueue::finishedTask() {
  m_taskChosen.store(false);
  return pickNextTask();
}

SerialTaskQueue::TaskBase* SerialTaskQueue::pickNextTask() {
  bool expect = false;
  if LIKELY (0 == m_pauseCount and m_taskChosen.compare_exchange_strong(expect, true)) {
    TaskBase* t = nullptr;
    if LIKELY (m_tasks.try_pop(t)) {
      return t;
    }
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

//
// const member functions
//

//
// static member functions
//
