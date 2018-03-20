// -*- C++ -*-
//
// Package:     FWCore/Framework
//
// Original Author:  W. David Dagenhart
//         Created:  1 December 2017

#include "FWCore/Framework/interface/stream/ProducingModuleHelper.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/stream/implementors.h"

namespace edm {
  namespace stream {

    void doAcquireIfNeeded(impl::ExternalWork* base,
                           Event const& ev,
                           EventSetup const& es,
                           WaitingTaskWithArenaHolder& holder) {
      base->acquire(ev, es, holder);
    }

    void doAcquireIfNeeded(void*,
                           Event const&,
                           EventSetup const&,
                           WaitingTaskWithArenaHolder&) {
    }
  }
}
