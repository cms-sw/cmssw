#ifndef FWCore_Concurrency_setNThreads_h
#define FWCore_Concurrency_setNThreads_h
//
//  setNThreads.h
//  CMSSW
//
//  Created by Chris Jones on 7/24/20.
//
#include <memory>
#include "FWCore/Concurrency/interface/ThreadsController.h"

namespace edm {
  unsigned int setNThreads(unsigned int iNThreads, unsigned int iStackSize, std::unique_ptr<ThreadsController>& oPtr);
}
#endif /* FWCore_Concurrency_setNThreads_h */
