#ifndef FWCore_Framework_ESTaskArenas_h
#define FWCore_Framework_ESTaskArenas_h

// -*- C++ -*-
//
// Package:     Framework
//

#include <tbb/task_arena.h>

namespace edm {
  tbb::task_arena& mainTaskArena();
  tbb::task_arena& esTaskArena();
}  // namespace edm

#endif /*FWCore_Framework_ESTaskArenas_h*/
