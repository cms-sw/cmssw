#ifndef FWCore_Framework_stream_ProducingModuleHelper_h
#define FWCore_Framework_stream_ProducingModuleHelper_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
//
// Original Author:  W. David Dagenhart
//         Created:  1 December 2017

namespace edm {

  class Event;
  class EventSetup;
  class WaitingTaskWithArenaHolder;

  namespace stream {

    namespace impl {
      class ExternalWork;
    }

    // Two overloaded functions, the first is called by doAcquire_
    // when the module inherits from ExternalWork. The first function
    // calls acquire, while the second function does nothing.
    void doAcquireIfNeeded(impl::ExternalWork*, Event const&, EventSetup const&, WaitingTaskWithArenaHolder&);

    void doAcquireIfNeeded(void*, Event const&, EventSetup const&, WaitingTaskWithArenaHolder&);
  }  // namespace stream
}  // namespace edm
#endif
