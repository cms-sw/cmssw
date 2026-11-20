#ifndef FWCore_ServiceRegistry_SystemBounds_h
#define FWCore_ServiceRegistry_SystemBounds_h
// -*- C++ -*-
//
// Package:     FWCore/ServiceRegistry
// Class  :     SystemBounds
//
/**\class SystemBounds SystemBounds.h "SystemBounds.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun, 08 Sep 2013 16:16:25 GMT
//

// system include files

// user include files

// forward declarations

#include <utility>
#include <vector>

#include "oneapi/tbb/task_arena.h"

namespace edm {
  namespace service {
    class SystemBounds {
    public:
      SystemBounds(unsigned int iNStreams,
                   unsigned int iNLumis,
                   unsigned int iNRuns,
                   unsigned int iNThreads,
                   std::vector<oneapi::tbb::task_arena*> iNumaArenas = {})
          : m_nStreams(iNStreams),
            m_nLumis(iNLumis),
            m_nRuns(iNRuns),
            m_nThreads(iNThreads),
            m_numaArenas(std::move(iNumaArenas)) {}

      // ---------- const member functions ---------------------
      unsigned int maxNumberOfStreams() const { return m_nStreams; }
      unsigned int maxNumberOfConcurrentRuns() const { return m_nRuns; }
      unsigned int maxNumberOfConcurrentLuminosityBlocks() const { return m_nLumis; }
      unsigned int maxNumberOfThreads() const { return m_nThreads; }
      // the TBB arenas of the NUMA domains in which the framework runs the events (process.options.numaLocality), or
      // none: services that observe the TBB threads (e.g. with a task_scheduler_observer) should observe them too
      std::vector<oneapi::tbb::task_arena*> const& numaArenas() const { return m_numaArenas; }

    private:
      // ---------- member data --------------------------------
      unsigned int m_nStreams;
      unsigned int m_nLumis;
      unsigned int m_nRuns;
      unsigned int m_nThreads;
      std::vector<oneapi::tbb::task_arena*> m_numaArenas;
    };

  }  // namespace service
}  // namespace edm

#endif
