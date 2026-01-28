#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <atomic>
#include <iostream>
#include <format>

namespace {
  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    MonitorAdaptor(unsigned int skip, size_t threshold) : m_skip{skip}, m_threshold{threshold} {}

  private:
    void allocCalled(size_t iRequested, size_t iActual, void const*) final {
      // returns previous value
      auto const before = m_presentActual.fetch_add(iActual, std::memory_order_acq_rel);
      auto const after = before + iActual;
      if (before < m_threshold and after >= m_threshold) {
        auto v = ++m_count;
        if (v > m_skip) {
          std::cout << std::format(
                           "abort threshold reached count: {} request: {} presentActual: {}", v, iRequested, after)
                    << std::endl;
          abort();
        }
      }
    }

    void deallocCalled(size_t iActual, void const*) final {
      if (0 == iActual)
        return;
      auto const present = m_presentActual.load(std::memory_order_acquire);
      if (present >= iActual) {
        m_presentActual.fetch_sub(iActual, std::memory_order_acq_rel);
      }
    }

    unsigned int m_skip;
    size_t m_threshold;
    std::atomic<unsigned int> m_count = 0;
    std::atomic<size_t> m_presentActual = 0;
  };
}  // namespace

class PresentThresholdAbortAllocMonitor {
public:
  PresentThresholdAbortAllocMonitor(edm::ParameterSet const& iPSet) {
    (void)cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>(
        iPSet.getUntrackedParameter<unsigned int>("skipCount"),
        iPSet.getUntrackedParameter<unsigned long long>("threshold"));
  };
  static void fillDescriptions(edm::ConfigurationDescriptions& iConfig) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<unsigned int>("skipCount", 0)
        ->setComment(
            "Number of times the present allocated memory crossing the thresholdmust be met before doing an abort. A "
            "value of 0 will happen the first "
            "time.");
    desc.addUntracked<unsigned long long>("threshold")
        ->setComment("Present allocated memory going over this limit triggers the abort. The units are bytes.");
    iConfig.addDefault(desc);
  }
};

typedef edm::serviceregistry::ParameterSetMaker<PresentThresholdAbortAllocMonitor>
    PresentThresholdAbortAllocMonitorMaker;
DEFINE_FWK_SERVICE_MAKER(PresentThresholdAbortAllocMonitor, PresentThresholdAbortAllocMonitorMaker);
