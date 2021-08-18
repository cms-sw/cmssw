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

// user include files
#include "FWCore/Concurrency/interface/TaskBase.h"

// forward declarations

namespace edm {
  template <typename F>
  class FunctorTask : public TaskBase {
  public:
    explicit FunctorTask(F f) : func_(std::move(f)) {}

    void execute() final { func_(); };

  private:
    F func_;
  };

  template <typename F>
  FunctorTask<F>* make_functor_task(F f) {
    return new FunctorTask<F>(std::move(f));
  }
}  // namespace edm

#endif
