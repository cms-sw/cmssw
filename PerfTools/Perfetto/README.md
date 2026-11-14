# PerfTools/Perfetto

In-process [Perfetto](https://perfetto.dev) tracing for `cmsRun`. The
`PerfettoTraceService` writes a `.pftrace` file that opens directly at
<https://ui.perfetto.dev>: a per-stream timeline of the framework transitions
(source, event, modules, `acquire`, EventSetup, cleanup), optionally with the
alpaka caching-allocator transactions, the CUDA kernels, CPU/GPU power, and
user-defined scopes.

The Perfetto SDK is the `perfetto` external (`<use name="perfetto"/>`).

## Contents

- `interface/CMSSWPerfettoCategories.h` — the `cmssw.*` track-event categories.
- `interface/CMSSWPerfettoLanes.h` — per-stream tracks and per-(stream, thread) lanes.
- `interface/CMSSWPerfettoModuleContext.h` — thread-local "current module", for attribution.
- `interface/CMSSWPerfettoTrace.h` — `CMS_PERFETTO_FUNC()` / `CMS_PERFETTO_SCOPE()`.
- `interface/PerfettoAllocatorMonitor.h` — caching-allocator → Perfetto bridge.
- `plugins/PerfettoCuptiProfiler.h` — CUDA kernels via CUPTI.
- `plugins/PerfettoPowerSampler.h` — CPU (RAPL) and GPU (NVML) power.
- `plugins/PerfettoTraceService.cc` — the service.
- `python/customisePerfetto.py` — `cmsDriver.py --customise` helper.
- `scripts/perfettoKernelResources.py` — static per-kernel resource usage (cuobjdump).

## Usage

```bash
cmsDriver.py step3 ... --customise PerfTools/Perfetto/customisePerfetto.customise
```

or in a configuration (parameters not given keep their defaults):

```python
from PerfTools.Perfetto.customisePerfetto import customisePerfetto
customisePerfetto(process, fileName="reco.pftrace", traceGpuKernels=True)
```

### Parameters (all untracked)

| parameter          | default          | meaning |
|--------------------|------------------|---------|
| `enabled`          | `True`           | master switch |
| `fileName`         | `cmsrun.pftrace` | output file, written at the end of the job |
| `bufferSizeKB`     | `262144`         | in-memory trace buffer; bounds the trace size |
| `shmemSizeKB`      | `65536`          | producer shared-memory buffer; if too small, slices are silently dropped |
| `maxEvents`        | `0`              | trace only the first N events (`0`: all) |
| `traceFunctions`   | `False`          | record the `CMS_PERFETTO_FUNC` / `CMS_PERFETTO_SCOPE` slices |
| `traceAllocations` | `False`          | record the alpaka caching-allocator transactions and memory counters |
| `traceGpuKernels`  | `False`          | record the CUDA kernels via CUPTI |
| `tracePower`       | `False`          | sample CPU (RAPL) and GPU (NVML) power |
| `powerPeriodMs`    | `1000`           | power sampling period |
| `traceModules`     | `[]`             | if not empty, trace only these module labels |

## Track layout

Within one event, independent modules run concurrently on different threads,
and an ExternalWork module's `acquire()` and `produce()` usually run on
different threads. Slices are therefore placed on a lane per (stream, thread):

```
process "cmsRun"
  ├─ edm::stream <sid>     "Event" slices + run/lumi/event counters
  │    └─ thread <n>       source / module / acquire / EventSetup / cleanup slices,
  │                        with the alloc/free instants and CMS_PERFETTO_FUNC slices nested
  ├─ GPU<d> stream <s>     CUDA kernels at their device-side times
  └─ counters              Throughput (events/s), memory (B), power (W)
```

A lane is fed by a single thread, so its slices nest correctly; concurrent work
of one stream shows up as parallel lanes. `Throughput (events/s)` is the event
rate over the last 16 completed events.

## Per-function tracing (`traceFunctions=True`)

```cpp
#include "PerfTools/Perfetto/interface/CMSSWPerfettoTrace.h"

void MyProducer::produce(edm::Event& event, edm::EventSetup const&) {
  CMS_PERFETTO_FUNC();  // or CMS_PERFETTO_SCOPE("name")
  ...
}
```

The slice nests under the module's slice. With `traceFunctions=False`, or
without the service, the macros cost one branch.

Avoid `using namespace cms::perfetto;` where the `TRACE_*` macros are used: it
makes their category lookup ambiguous.

## Caching-allocator tracing (`traceAllocations=True`)

The alpaka `CachingAllocator` calls an optional, process-wide
`cms::alpakatools::CachingAllocatorMonitor` on every transaction; without one it
costs one atomic load. The service installs `PerfettoAllocatorMonitor`, which
records

- every `alloc` / `free` as an instant under the slice of the module that made
  it, annotated with the module, the bin-rounded and requested sizes, cache
  hit/miss, device and queue;
- the `live`, `cached` and `requested` byte totals of each memory space
  (`dev<d>`, and `host` for pinned memory) as counters.

A freed block may still be used by queued device work: the allocator re-hands it
only once an event recorded on its queue has completed.

## GPU kernel tracing (`traceGpuKernels=True`)

[CUPTI](https://docs.nvidia.com/cupti/) activity tracing records every CUDA
kernel, including those from release plugins, without serializing them. Each
kernel is a slice at its device-side start and end, annotated with
`registers_per_thread`, `static_smem_B`, `dynamic_smem_B`, `local_per_thread_B`,
`local_total_B`, `grid`, `block`, an `occupancy_est` (thread, register and
shared-memory limits only), the full name and the CUPTI `correlation_id`.
Without CUDA the plugin builds a no-op.

`scripts/perfettoKernelResources.py` is the static counterpart: it runs
`cuobjdump --dump-resource-usage` on built libraries and also reports stack size
and register spills:

```bash
perfettoKernelResources.py --filter Phase2 \
  $CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH/pluginRecoLocalTrackerSiPixelClusterizerPluginsPortableCudaAsync.so
```

## Nested parallelism

- A module's slice covers its wall-clock time on its thread. A module blocked in
  a non-isolated `tbb::parallel_for` can run tasks of other modules meanwhile:
  they appear nested in its slice and lengthen it. The module context is a stack,
  so attribution is restored when they return.
- Threads helping in a `parallel_for` do not inherit the module context. Wrap the
  body to attribute their allocations and scopes to the module:

  ```cpp
  tbb::parallel_for(range, cms::perfetto::withModuleContext([&](auto const& r) { ... }));
  ```

## Profiling one algorithm

```python
customisePerfetto(process,
                  fileName="myalgo.pftrace",
                  traceModules=["myProducerAlpaka"],
                  traceGpuKernels=True,
                  traceAllocations=True)
```

Things to look for:

- an asynchronous GPU module has a short host slice, while its cost is on the
  `GPU<d> stream <s>` tracks;
- a low `occupancy_est` with many `registers_per_thread` or much shared memory
  means an occupancy-limited kernel;
- `alloc` with `cache_hit=false` is a real device allocation; few cache hits
  after the first events means the cache is not absorbing the churn;
- the gap between the `live` and `requested` counters is the bin rounding;
- an `acquire`, a gap, then `produce` on another thread is the normal
  ExternalWork pattern: the gap is the device work.

## Overhead

Measured with FastTimerService on the Phase-2 HLT at PU 200 (8 threads, 500
events): below 1% of CPU time with every option enabled, within run-to-run noise
for the default configuration. `traceModules` reduces it further.
