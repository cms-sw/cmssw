// -*- C++ -*-
//
// Package:     FWCore/Framework
//
// Original Author:  W. David Dagenhart
//         Created:  1 December 2017

#include "FWCore/Framework/interface/stream/ProducingModuleHelper.h"
#include "FWCore/Framework/interface/stream/implementors.h"

namespace edm {
  namespace stream {

    void doAcquireIfNeeded(impl::ExternalWork* base, Event const& ev, EventSetup const& es, WaitingTaskHolder&& holder) {
      base->acquire(ev, es, WaitingTaskWithArenaHolder(std::move(holder)));
    }

    void doAcquireIfNeeded(void*, Event const&, EventSetup const&, WaitingTaskHolder&&) {}
  }  // namespace stream
}  // namespace edm
