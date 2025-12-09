#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
// For scoped locks
#include <mutex>

#include <oneapi/tbb/concurrent_vector.h>

#include <fmt/printf.h>

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvtx3.hpp>

#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"

#include "FWCore/Services/interface/ProfilerService.h"

namespace {
  /**
   * \brief Backend for NVidia's Nsight Systems profiling.
   */
  class NVTXBackend {
  public:
    // Forward definitions
    using Color = ProfilerServiceBase::Color;
    static constexpr auto to_underlying = ProfilerServiceBase::to_underlying;
    using SpinLock = ProfilerServiceBase::SpinLock;
    class Range;
    class Domain;
    static void mark(const Domain& domain, const char* message, Color color);
    /**
     * \note These functions can be used multiple times. See doc at:
     * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html
     */
    static void profilerStart() { cudaProfilerStart(); }
    static void profilerStop() { cudaProfilerStop(); }

  private:
    static __attribute__((unused)) void nvtxDomainMarkColor(nvtxDomainHandle_t domain,
                                                            const char* message,
                                                            uint32_t color) {
      nvtxEventAttributes_t eventAttrib = {};
      eventAttrib.version = NVTX_VERSION;
      eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      eventAttrib.colorType = NVTX_COLOR_ARGB;
      eventAttrib.color = color;
      eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
      eventAttrib.message.ascii = message;
      nvtxDomainMarkEx(domain, &eventAttrib);
    }

  public:
    using EDMService = edm::Service<CUDAInterface>;
    class Domain {
    public:
      friend class Range;
      friend void NVTXBackend::mark(const Domain& domain, const char* message, Color color);
      Domain() = default;
      ~Domain() {
        if (domain_ != nvtxInvalidDomainId) {
          nvtxDomainDestroy(domain_);
          domain_ = nvtxInvalidDomainId;
        }
      }
      void create(const std::string& name) {
        // Assert the domain is created only once
        assert(domain_ == nvtxInvalidDomainId);
        domain_ = nvtxDomainCreateA(name.c_str());
      }
      void destroy() {
        if (domain_ != nvtxInvalidDomainId) {
          nvtxDomainDestroy(domain_);
          domain_ = nvtxInvalidDomainId;
        }
      }

    private:
      static constexpr nvtxDomainHandle_t nvtxInvalidDomainId = nullptr;
      nvtxDomainHandle_t nativeHandle() const { return domain_; }
      nvtxDomainHandle_t domain_ = nvtxInvalidDomainId;
    };

    class Range {
    public:
      friend void NVTXBackend::mark(const Domain& domain, const char* message, Color color);
      Range() = default;
      // copy constructor deleted
      Range(const Range&) = delete;
      /// Move copy constructor: we take a lock and move the contents
      /// We need it to resize vectors
      Range(Range&& o) noexcept {
        std::scoped_lock lock(o.mtx_);
        std::scoped_lock lock2(mtx_);
        domain_ = o.domain_;
        range_ = o.range_;
        o.domain_ = Domain::nvtxInvalidDomainId;
        o.range_ = nvtxInvalidRangeId;
      }
      ~Range() {
        std::scoped_lock lock(mtx_);
        if (range_ != nvtxInvalidRangeId)
          nvtxDomainRangeEnd(domain_, range_);
      }

    private:
      nvtxRangeId_t nvtxDomainRangeStartColor(const Domain& domain, const char* message, uint32_t color) {
        nvtxEventAttributes_t eventAttrib = {};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = color;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = message;
        return nvtxDomainRangeStartEx(domain.nativeHandle(), &eventAttrib);
      }

      static constexpr nvtxRangeId_t nvtxInvalidRangeId = ~0ul;

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
#undef NVTX_RANGE_DEBUG
#ifdef NVTX_RANGE_DEBUG
        std::fprintf(stderr,
                     "[NVTX_RANGE] start this=%p domain=%p range=%lu msg=%s where=%s\n",
                     static_cast<const void*>(this),
                     static_cast<const void*>(domain_),
                     static_cast<unsigned long>(range_),
                     message ? message : "(null)",
                     where ? where : "(null)");
#endif
        if (range_ != nvtxInvalidRangeId) {
          std::string fullmsg =
              fmt::sprintf("Warning: previous range not ended before starting a new one in %s for %s", where, message);
          abort();
          nvtxDomainMarkColor(domain_, fullmsg.c_str(), colorMap[to_underlying(Color::Red)]);
          nvtxDomainRangeEnd(domain_, range_);
        }
        domain_ = domain.nativeHandle();
        range_ = nvtxDomainRangeStartColor(domain, message, colorMap[to_underlying(color)]);
      }

      void endIn(const Domain& domain, const char* message, const char* where) {
        std::scoped_lock lock(mtx_);
#ifdef NVTX_RANGE_DEBUG
        std::fprintf(stderr,
                     "[NVTX_RANGE] end   this=%p domain=%p range=%lu msg=%s where=%s\n",
                     static_cast<const void*>(this),
                     static_cast<const void*>(domain_),
                     static_cast<unsigned long>(range_),
                     message ? message : "(null)",
                     where ? where : "(null)");
#endif
        if (range_ != nvtxInvalidRangeId) {
          nvtxDomainRangeEnd(domain_, range_);
          range_ = nvtxInvalidRangeId;
          domain_ = Domain::nvtxInvalidDomainId;
        } else {
          std::string fullmsg =
              fmt::sprintf("Warning: trying to end a range that is not started in %s for %s", where, message);
          abort();
          nvtxDomainMarkColor(domain.nativeHandle(), fullmsg.c_str(), colorMap[to_underlying(Color::Red)]);
        }
      }

    private:
      nvtxRangeId_t range_ = nvtxInvalidRangeId;
      nvtxDomainHandle_t domain_ = Domain::nvtxInvalidDomainId;
      SpinLock mtx_ = SpinLock{};
    };

    static std::string shortName() { return "NV"; }
    static std::string serviceComment() {
      return R"(This Service provides CMSSW-aware annotations to nvprof/nvvm.

Notes on nvprof options:
  - the option '--profile-from-start off' should be used if skipFirstEvent is True.
 - the option '--cpu-profiling on' currently results in cmsRun being stuck at the beginning of the job.
 - the option '--cpu-thread-tracing on' is not compatible with jemalloc, and should only be used with cmsRunGlibC.)";
    }
  };

  void NVTXBackend::mark(const NVTXBackend::Domain& domain, const char* message, Color color) {
    NVTXBackend::nvtxDomainMarkColor(
        domain.nativeHandle(), message, NVTXBackend::Range::colorMap[to_underlying(color)]);
  }
}  // namespace

class NVProfilerService : public ProfilerService<NVTXBackend> {
public:
  using ProfilerService<NVTXBackend>::ProfilerService;
};

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(NVProfilerService);
