#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

#include <atomic>
#include <string_view>

namespace {
  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    void activate() { active_ = true; }
    void deactivate() { active_ = false; }

    void reportAndReset(std::string_view name) {
      deactivate();
      auto const requested = requested_.exchange(0, std::memory_order_acq_rel);
      auto const nAllocations = nAllocations_.exchange(0, std::memory_order_acq_rel);
      auto const nDeallocations = nDeallocations_.exchange(0, std::memory_order_acq_rel);
      auto const maxSingleRequested = maxSingleRequested_.exchange(0, std::memory_order_acq_rel);

      // presentActual needs special treatment to find the added
      auto const presentActual = presentActual_.load(std::memory_order_acquire);
      auto const previousPresentActual = previousPresentActual_.exchange(presentActual, std::memory_order_acq_rel);
      auto const added = presentActual - previousPresentActual;

      // use the presentActual as the starting point for the maximum of the next measurement
      auto const maxActual = maxActual_.exchange(presentActual, std::memory_order_acq_rel);

      edm::LogAbsolute("PhaseAllocMonitor")
          .format(
              "PhaseAllocMonitor> {}: requested {} max alloc {} presentActual {} peak {} added {} nAlloc {} nDealloc "
              "{}",
              name,
              requested,
              maxSingleRequested,
              presentActual,
              maxActual,
              added,
              nAllocations,
              nDeallocations);

      activate();
    }

  private:
    void allocCalled(size_t iRequested, size_t iActual, void const*) final {
      if (not active_) {
        return;
      }

      nAllocations_.fetch_add(1, std::memory_order_acq_rel);
      requested_.fetch_add(iRequested, std::memory_order_acq_rel);

      //returns previous value
      auto a = presentActual_.fetch_add(iActual, std::memory_order_acq_rel);
      a += iActual;

      auto max = maxActual_.load(std::memory_order_relaxed);
      while (a > max) {
        if (maxActual_.compare_exchange_strong(max, a, std::memory_order_acq_rel)) {
          break;
        }
      }

      auto single = maxSingleRequested_.load(std::memory_order_relaxed);
      while (iRequested > single) {
        if (maxSingleRequested_.compare_exchange_strong(single, iRequested, std::memory_order_acq_rel)) {
          break;
        }
      }
    }
    void deallocCalled(size_t iActual, void const*) final {
      if (not active_) {
        return;
      }

      if (0 == iActual)
        return;
      nDeallocations_.fetch_add(1, std::memory_order_acq_rel);
      presentActual_.fetch_sub(iActual, std::memory_order_acq_rel);
    }

    std::atomic<bool> active_ = false;
    std::atomic<size_t> requested_ = 0;
    std::atomic<long long> previousPresentActual_ = 0;
    std::atomic<long long> presentActual_ = 0;
    std::atomic<long long> maxActual_ = 0;
    std::atomic<size_t> nAllocations_ = 0;
    std::atomic<size_t> nDeallocations_ = 0;
    std::atomic<size_t> maxSingleRequested_ = 0;
  };

}  // namespace

class PhaseAllocMonitor {
public:
  PhaseAllocMonitor(edm::ParameterSet const& iPS, edm::ActivityRegistry& iAR) {
    auto const showSignals = iPS.getUntrackedParameter<std::vector<std::string>>("showSignals");
    auto adaptor = cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();

    iAR.watchPostServicesConstruction([adaptor]() {
      adaptor->activate();
      adaptor->reportAndReset("PostServicesConstruction");
    });

#define REGISTER(SIGNAL, ...)                                                                              \
  if (showSignals.empty() or std::ranges::find(showSignals, EDM_STRINGIZE(SIGNAL)) != showSignals.end()) { \
    iAR.watch##SIGNAL([adaptor](__VA_ARGS__) { adaptor->reportAndReset(EDM_STRINGIZE(SIGNAL)); });         \
  }

    REGISTER(PreEventSetupModulesConstruction);
    REGISTER(PostEventSetupModulesConstruction);
    REGISTER(PreModulesAndSourceConstruction);
    REGISTER(PostModulesAndSourceConstruction);

    REGISTER(PreFinishSchedule);
    REGISTER(PostFinishSchedule);
    REGISTER(PrePrincipalsCreation);
    REGISTER(PostPrincipalsCreation);

    REGISTER(PreEventSetupConfigurationFinalized);
    REGISTER(PostEventSetupConfigurationFinalized);
    REGISTER(PreBeginJob, edm::ProcessContext const&);
    REGISTER(PostBeginJob);

    // Note that the first {Pre,Post}OpenFile is issued from the
    // Source construction concurrently to the EDModule construction.
    // Therefore, in a multithreaded the memory counters from the
    // first signals might not be reliable. The subsequent signals are
    // synchronization points for the framework.
    REGISTER(PreOpenFile, std::string const&);
    REGISTER(PostOpenFile, std::string const&);

    REGISTER(PreCloseFile, std::string const&);
    REGISTER(PostCloseFile, std::string const&);

    REGISTER(PreOpenOutputFiles);
    REGISTER(PostOpenOutputFiles);
    REGISTER(PreCloseOutputFiles);
    REGISTER(PostCloseOutputFiles);

    REGISTER(JobFailure);
    REGISTER(BeginProcessing);
    REGISTER(EndProcessing);

    REGISTER(PreEndJob);
    REGISTER(PostEndJob);

#undef REGISTER
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
    edm::ParameterSetDescription ps;
    ps.addUntracked<std::vector<std::string>>("showSignals", {})
        ->setComment(
            "Show measurements for these ActivityRegistry signals. Empty value means show all signals. For the list of "
            "possible signals see the code.");
    iDesc.addDefault(ps);
  }
};

DEFINE_FWK_SERVICE(PhaseAllocMonitor);
