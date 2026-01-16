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
    MonitorAdaptor(unsigned int skip, size_t minThreshold, size_t maxThreshold)
        : m_skip{skip}, m_min{minThreshold}, m_max{maxThreshold}, m_count{0} {}

  private:
    void allocCalled(size_t iRequested, size_t iActual, void const*) final {
      if (iRequested >= m_min) {
        if (m_max == 0 or iRequested <= m_max) {
          auto v = ++m_count;
          if (v > m_skip) {
            std::cout << std::format("abort threshold reached count: {} request: {}", v, iRequested) << std::endl;
            abort();
          }
        }
      }
    }
    void deallocCalled(size_t iActual, void const*) final {}

    unsigned int m_skip;
    size_t m_min;
    size_t m_max;
    std::atomic<unsigned int> m_count;
  };
}  // namespace

class ThresholdAbortAllocMonitor {
public:
  ThresholdAbortAllocMonitor(edm::ParameterSet const& iPSet) {
    (void)cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>(
        iPSet.getUntrackedParameter<unsigned int>("skipCount"),
        iPSet.getUntrackedParameter<unsigned long long>("minThreshold"),
        iPSet.getUntrackedParameter<unsigned long long>("maxThreshold"));
  };
  static void fillDescriptions(edm::ConfigurationDescriptions& iConfig) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<unsigned int>("skipCount", 0)
        ->setComment(
            "Number of times the threshold range must be met before doing an abort. A value of 0 will happen the first "
            "time.");
    desc.addUntracked<unsigned long long>("minThreshold")
        ->setComment(
            "A memory request below this value will not trigger the abort. The exact value is allowed to trigger the "
            "abort. Triggering the abort also requires being within 'maxThreshold' setting. The units are bytes.");
    desc.addUntracked<unsigned long long>("maxThreshold", 0)
        ->setComment(
            "A memory request above this value will not trigger the abort. A value of 0 means any value could trigger "
            "the abort. The exact value is allowed to trigger the abort. Triggering the abort also depends on being "
            "within 'minThreshold' setting. The units are bytes.");
    iConfig.addDefault(desc);
  }
};

typedef edm::serviceregistry::ParameterSetMaker<ThresholdAbortAllocMonitor> ThresholdAbortAllocMonitorMaker;
DEFINE_FWK_SERVICE_MAKER(ThresholdAbortAllocMonitor, ThresholdAbortAllocMonitorMaker);
