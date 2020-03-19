#ifndef FWCore_Concurrency_FunctorTask_h
#define FWCore_Concurrency_FunctorTask_h
// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     FunctorTask
//
/**\class FunctorTask FunctorTask.h FWCore/Concurrency/interface/FunctorTask.h

 Description: Builds a tbb::task from a lambda.

 Usage:
 
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 13:46:31 CST 2013
// $Id$
//

// system include files
#include <atomic>
#include <exception>
#include <memory>
#include "tbb/task.h"

// user include files

// forward declarations

namespace edm {
  template <typename F>
  class FunctorTask : public tbb::task {
  public:
    explicit FunctorTask(F f) : func_(std::move(f)) {}

    task* execute() override {
      func_();
      return nullptr;
    };

  private:
    F func_;
  };

  template <typename ALLOC, typename F>
  FunctorTask<F>* make_functor_task(ALLOC&& iAlloc, F f) {
    return new (iAlloc) FunctorTask<F>(std::move(f));
  }
}  // namespace edm

#endif
