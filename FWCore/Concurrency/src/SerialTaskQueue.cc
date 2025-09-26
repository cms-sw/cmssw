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
#include "FWCore/Utilities/interface/make_sentry.h"

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
  if UNLIKELY (0 != m_pauseCount)
    return nullptr;
  bool expect = false;
  //need pop task and setting m_taskChosen to be atomic to avoid
  // case where thread pauses just after try_pop failed but then
  // a task is added and that call fails the check on m_taskChosen
  while (not m_pickingNextTask.compare_exchange_strong(expect, true)) {
    expect = false;
  }
  auto sentry = edm::make_sentry(&m_pickingNextTask, [](auto* v) { v->store(false); });

  if LIKELY (m_taskChosen.compare_exchange_strong(expect, true)) {
    TaskBase* t = nullptr;
    if LIKELY (m_tasks.try_pop(t)) {
      return t;
    }
    //no task was actually pulled
    m_taskChosen.store(false);
  }
  return nullptr;
}

//
// const member functions
//

//
// static member functions
//
