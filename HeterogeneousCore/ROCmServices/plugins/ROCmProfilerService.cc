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

#include "HeterogeneousCore/ROCmServices/interface/ROCmInterface.h"

#include "PerfTools/ProfilerService/interface/ProfilerService.h"

namespace {
  /**
   * \brief Backend for ROCm profiling.
   * \note All APIs used are part of ROCTX. See documentation at 
   * https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofiler-sdk-roctx.html
   */
  class ROCmBackend {
  public:
    // Forward definitions
    using Color = ProfilerServiceBase::Color;
    static constexpr auto to_underlying = ProfilerServiceBase::to_underlying;
    using SpinLock = ProfilerServiceBase::SpinLock;
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

      static constexpr std::array<uint32_t, 32> colorMap = {{
          0x00000000,  // Black
          // Red family
          0x00880000,  // Red_Dark2
          0x00cc0000,  // Red_Dark1
          0x00ff0000,  // Red
          0x00ff8080,  // Red_Light1
          0x00ffcccc,  // Red_Light2
          // Green family
          0x00004400,  // Green_Dark2
          0x00009900,  // Green_Dark1
          0x0000ff00,  // Green
          0x0099ff99,  // Green_Light1
          0x00ccffcc,  // Green_Light2
          // Blue family
          0x00000077,  // Blue_Dark2
          0x000000bb,  // Blue_Dark1
          0x000000ff,  // Blue
          0x009999ff,  // Blue_Light1
          0x00ccccff,  // Blue_Light2
          // Amber family
          0x00886600,  // Amber_Dark2
          0x00cc9900,  // Amber_Dark1
          0x00ffbf00,  // Amber
          0x00ffd966,  // Amber_Light1
          0x00fff2cc,  // Amber_Light2
          0x00ffffff,  // White
          // Grey family
          0x00404040,  // Grey_Dark2
          0x00606060,  // Grey_Dark1
          0x00808080,  // Grey
          0x00a0a0a0,  // Grey_Light1
          0x00c0c0c0,  // Grey_Light2
          // Yellow family
          0x00888800,  // Yellow_Dark2
          0x00cccc00,  // Yellow_Dark1
          0x00ffff00,  // Yellow
          0x00ffff66,  // Yellow_Light1
          0x00ffffcc   // Yellow_Light2
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
