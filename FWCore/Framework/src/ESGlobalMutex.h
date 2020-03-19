#ifndef FWCore_Framework_ESGlobalMutex_h
#define FWCore_Framework_ESGlobalMutex_h

// -*- C++ -*-
//
// Package:     Framework
//
/**
Description: This should only be used by the parts of the Framework
supporting the EventSetup system.  It is used in the functions:

    edm::eventsetup::DataProxy::get
    edm::EventSetupRecordIntervalFinder::findIntervalFor

This protects activity that can't be safely run concurrently.
For example, database transactions in CondDBESSource (aka
PoolDBESSource).

The current intent is that the existence of this mutex is temporary.
In the future, the need for it will be replaced by use of things like
WaitingTaskLists and WaitingTaskHolders used in such a way that this
can be handled in a lock-free manner.
*/
// Author: W. David Dagenhart
// Created: 9 August 2019

#include <mutex>

namespace edm {

  std::recursive_mutex& esGlobalMutex();

}  // namespace edm
#endif  // FWCore_Framework_ESGlobalMutex_h
