// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     WaitingTaskList
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 13:46:45 CST 2013
// $Id$
//

// system include files

// user include files
#include "tbb/task.h"
#include <cassert>
#include <memory>

#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/hardware_pause.h"
#include "FWCore/Utilities/interface/Likely.h"

using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
WaitingTaskList::WaitingTaskList(unsigned int iInitialSize)
    : m_head{nullptr},
      m_nodeCache{new WaitNode[iInitialSize]},
      m_nodeCacheSize{iInitialSize},
      m_lastAssignedCacheIndex{0},
      m_waiting{true} {
  auto nodeCache = m_nodeCache.get();
  for (auto it = nodeCache, itEnd = nodeCache + m_nodeCacheSize; it != itEnd; ++it) {
    it->m_fromCache = true;
  }
}

//
// member functions
//
void WaitingTaskList::reset() {
  m_exceptionPtr = std::exception_ptr{};
  unsigned int nSeenTasks = m_lastAssignedCacheIndex;
  m_lastAssignedCacheIndex = 0;
  assert(m_head == nullptr);
  if (nSeenTasks > m_nodeCacheSize) {
    //need to expand so next time we don't have to do any
    // memory requests
    m_nodeCacheSize = nSeenTasks;
    m_nodeCache = std::make_unique<WaitNode[]>(nSeenTasks);
    auto nodeCache = m_nodeCache.get();
    for (auto it = nodeCache, itEnd = nodeCache + m_nodeCacheSize; it != itEnd; ++it) {
      it->m_fromCache = true;
    }
  }
  //this will make sure all cores see the changes
  m_waiting = true;
}

WaitingTaskList::WaitNode* WaitingTaskList::createNode(WaitingTask* iTask) {
  unsigned int index = m_lastAssignedCacheIndex++;

  WaitNode* returnValue;
  if (index < m_nodeCacheSize) {
    returnValue = m_nodeCache.get() + index;
  } else {
    returnValue = new WaitNode;
    returnValue->m_fromCache = false;
  }
  returnValue->m_task = iTask;
  //No other thread can see m_next yet. The caller to create node
  // will be doing a synchronization operation anyway which will
  // make sure m_task and m_next are synched across threads
  returnValue->m_next.store(returnValue, std::memory_order_relaxed);

  return returnValue;
}

void WaitingTaskList::add(WaitingTaskHolder iTask) {
  if (!m_waiting) {
    if (m_exceptionPtr) {
      iTask.doneWaiting(m_exceptionPtr);
    }
  } else {
    auto task = iTask.release_no_decrement();
    WaitNode* newHead = createNode(task);
    //This exchange is sequentially consistent thereby
    // ensuring ordering between it and setNextNode
    WaitNode* oldHead = m_head.exchange(newHead);
    newHead->setNextNode(oldHead);

    //For the case where oldHead != nullptr,
    // even if 'm_waiting' changed, we don't
    // have to recheck since we beat 'announce()' in
    // the ordering of 'm_head.exchange' call so iTask
    // is guaranteed to be in the link list

    if (nullptr == oldHead) {
      newHead->setNextNode(nullptr);
      if (!m_waiting) {
        //if finished waiting right before we did the
        // exchange our task will not be spawned. Also,
        // additional threads may be calling add() and swapping
        // heads and linking us to the new head.
        // It is safe to call announce from multiple threads
        announce();
      }
    }
  }
}

void WaitingTaskList::add(WaitingTask* iTask) {
  iTask->increment_ref_count();
  if (!m_waiting) {
    if (UNLIKELY(bool(m_exceptionPtr))) {
      iTask->dependentTaskFailed(m_exceptionPtr);
    }
    if (0 == iTask->decrement_ref_count()) {
      tbb::task::spawn(*iTask);
    }
  } else {
    WaitNode* newHead = createNode(iTask);
    //This exchange is sequentially consistent thereby
    // ensuring ordering between it and setNextNode
    WaitNode* oldHead = m_head.exchange(newHead);
    newHead->setNextNode(oldHead);

    //For the case where oldHead != nullptr,
    // even if 'm_waiting' changed, we don't
    // have to recheck since we beat 'announce()' in
    // the ordering of 'm_head.exchange' call so iTask
    // is guaranteed to be in the link list

    if (nullptr == oldHead) {
      if (!m_waiting) {
        //if finished waiting right before we did the
        // exchange our task will not be spawned. Also,
        // additional threads may be calling add() and swapping
        // heads and linking us to the new head.
        // It is safe to call announce from multiple threads
        announce();
      }
    }
  }
}

void WaitingTaskList::presetTaskAsFailed(std::exception_ptr iExcept) {
  if (iExcept and m_waiting) {
    WaitNode* node = m_head.load();
    while (node) {
      WaitNode* next;
      while (node == (next = node->nextNode())) {
        hardware_pause();
      }
      node->m_task->dependentTaskFailed(iExcept);
      node = next;
    }
  }
}

void WaitingTaskList::announce() {
  //Need a temporary storage since one of these tasks could
  // cause the next event to start processing which would refill
  // this waiting list after it has been reset
  WaitNode* n = m_head.exchange(nullptr);
  WaitNode* next;
  while (n) {
    //it is possible that 'WaitingTaskList::add' is running in a different
    // thread and we have a new 'head' but the old head has not yet been
    // attached to the new head (we identify this since 'nextNode' will return itself).
    //  In that case we have to wait until the link has been established before going on.
    while (n == (next = n->nextNode())) {
      hardware_pause();
    }
    auto t = n->m_task;
    if (UNLIKELY(bool(m_exceptionPtr))) {
      t->dependentTaskFailed(m_exceptionPtr);
    }
    if (!n->m_fromCache) {
      delete n;
    }
    n = next;

    //the task may indirectly call WaitingTaskList::reset
    // so we need to call spawn after we are done using the node.
    if (0 == t->decrement_ref_count()) {
      tbb::task::spawn(*t);
    }
  }
}

void WaitingTaskList::doneWaiting(std::exception_ptr iPtr) {
  m_exceptionPtr = iPtr;
  m_waiting = false;
  announce();
}
