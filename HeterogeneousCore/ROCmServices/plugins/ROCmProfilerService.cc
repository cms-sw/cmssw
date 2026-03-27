#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
// For scoped locks
#include <mutex>

#include <oneapi/tbb/concurrent_vector.h>

#include <fmt/printf.h>

#include <rocprofiler-sdk-roctx/roctx.h>

#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "HeterogeneousCore/ROCmServices/interface/ROCmInterface.h"

#include "FWCore/Services/interface/ProfilerService.h"

namespace {
  /**
   * \brief Spinlock mutex for thread safety without returning to kernel.
   */
  class SpinLock {
  public:
    SpinLock() : flag_(ATOMIC_FLAG_INIT) {}
    void lock() {
      while (flag_.test_and_set(std::memory_order_acquire))
        ;
    }
    void unlock() { flag_.clear(std::memory_order_release); }

  private:
    std::atomic_flag flag_;
  };

  /**
   * \brief Backend for ROCm profiling.
   * \note All APIs used are part of ROCTX. See documentation at 
   * https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofiler-sdk-roctx.html
   */
  class ROCmBackend {
  public:
    // Forward definitions
    using Color = ProfilerServiceColor;
    class Range;
    class Domain;
    static void mark(const Domain& domain, const char* message, Color color);
    /**
     * \note Latest doc is broken at time of writing, but older version is useful:
     * https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.0.2/api-reference/rocprofiler-sdk-roctx_api/roctx_modules/profiler-control.html
     */
    static void profilerStart() {
      // 0 for all threads.
      roctxProfilerResume(0);
    }
    static void profilerStop() {
      // 0 for all threads.
      roctxProfilerPause(0);
    }

  public:
    using EDMService = edm::Service<ROCmInterface>;
    class Domain {
    public:
      friend class Range;
      friend void ROCmBackend::mark(const Domain& domain, const char* message, Color color);
      Domain() = default;
      ~Domain() = default;
      void create(const std::string& name) {
        // Assert the domain is created only once
        assert(domain_.empty());
        domain_ = name;
      }
      void destroy() { domain_.clear(); }

    private:
      const std::string& nativeHandle() const { return domain_; }
      std::string domain_;
    };

    class Range {
    public:
      friend void ROCmBackend::mark(const Domain& domain, const char* message, Color color);
      Range() = default;
      // copy constructor deleted
      Range(const Range&) = delete;
      /// Move copy constructor: we take a lock and move the contents
      /// We need it to resize vectors of unique_range_in
      Range(Range&& o) noexcept {
        std::scoped_lock lock(o.mtx_);
        std::scoped_lock lock2(mtx_);
        domain_ = o.domain_;
        range_ = o.range_;
        o.domain_.clear();
        o.range_ = roctxInvalidRangeId;
      }
      ~Range() {
        std::scoped_lock lock(mtx_);
        if (range_ != roctxInvalidRangeId)
          roctxRangeStop(range_);
      }

    private:
      roctx_range_id_t roctxDomainRangeStartColor(const Domain& domain, const char* message, uint32_t color) {
        return roctxRangeStartA((domain.nativeHandle() + "-" + message).c_str());
      }

      static constexpr roctx_range_id_t roctxInvalidRangeId = ~0ul;

      static constexpr std::array<uint32_t, 9> colorMap = {{
          0x00000000,  // Black
          0x00ff0000,  // Red
          0x00009900,  // Dark Green
          0x0000ff00,  // Green
          0x00ccffcc,  // Light Green
          0x000000ff,  // Blue
          0x00ffbf00,  // Amber
          0x00fff2cc,  // Light Amber
          0x00ffffff   // White
      }};

    public:
      void startColorIn(const Domain& domain, const char* message, Color color, const char* where) {
        std::scoped_lock lock(mtx_);
        if (range_ != roctxInvalidRangeId) {
          std::string fullmsg =
              fmt::sprintf("Warning: previous range not ended before starting a new one in %s for %s", where, message);
          roctxMarkA((domain_ + "-" + fullmsg).c_str());
          roctxRangeStop(range_);
        }
        domain_ = domain.nativeHandle();
        range_ = roctxRangeStartA((domain_ + "-" + message).c_str());
      }

      void endIn(const Domain& domain, const char* message, const char* where) {
        std::scoped_lock lock(mtx_);
        if (range_ != roctxInvalidRangeId) {
          roctxRangeStop(range_);
          range_ = roctxInvalidRangeId;
          domain_.clear();
        } else {
          std::string fullmsg =
              fmt::sprintf("Warning: trying to end a range that is not started in %s for %s", where, message);
          roctxMarkA((domain_ + "-" + fullmsg).c_str());
        }
      }

    private:
      roctx_range_id_t range_ = roctxInvalidRangeId;
      std::string domain_;
      SpinLock mtx_ = SpinLock{};
    };

    static std::string shortName() { return "ROCm"; }
    static std::string serviceComment() {
      return R"(This Service provides CMSSW-aware annotations to nvprof/nvvm.

Notes on nvprof options:
  - the option '--profile-from-start off' should be used if skipFirstEvent is True.
 - the option '--cpu-profiling on' currently results in cmsRun being stuck at the beginning of the job.
 - the option '--cpu-thread-tracing on' is not compatible with jemalloc, and should only be used with cmsRunGlibC.)";
    }
  };

  void ROCmBackend::mark(const ROCmBackend::Domain& domain, const char* message, Color color) {
    roctxMark(("[" + domain.nativeHandle() + "]: " + std::string(message)).c_str());
  }
}  // namespace

class ROCmProfilerService : public ProfilerService<ROCmBackend> {
public:
  using ProfilerService<ROCmBackend>::ProfilerService;
};

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(ROCmProfilerService);
