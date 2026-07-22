# PerfTools/Perfetto

In-process [Perfetto](https://perfetto.dev) tracing for `cmsRun`. The
`PerfettoTraceService` records a `.pftrace` file that can be opened directly at
<https://ui.perfetto.dev>, giving a per-stream timeline of the framework phases
(source read, event lifetime, module execution, `acquire`, EventSetup, cleanup).

## Contents

The Perfetto SDK comes from the `perfetto` CMSSW external (`<use name="perfetto"/>`,
`#include <perfetto.h>`), not vendored here.

- `interface/CMSSWPerfettoCategories.h` — the `cmssw.*` track-event categories.
- `interface/CMSSWPerfettoModuleContext.{h,cc}` — thread-local "current module",
  set by the service so the allocator/GPU layers can attribute their work.
- `interface/CMSSWPerfettoTrace.h` — `CMS_PERFETTO_FUNC()` / `CMS_PERFETTO_SCOPE()`
  scoped-slice macros for optional intra-module instrumentation.
- `interface/PerfettoAllocatorMonitor.h` — caching-allocator → Perfetto bridge.
- `plugins/PerfettoCuptiProfiler.h` — CUPTI → Perfetto GPU kernel tracing.
- `plugins/PerfettoPowerSampler.h` — CPU (RAPL) + GPU (NVML) power sampling.
- `plugins/PerfettoTraceService.cc` — the EDM service.
- `python/customisePerfetto.py` — `cmsDriver.py --customise` helper.
- `scripts/perfettoKernelResources.py` — static (ptxas/cuobjdump) kernel resource dump.

## Track layout

CMSSW runs one global TBB arena; within a *single* stream and a *single* event,
independent modules execute concurrently on different threads, and an ExternalWork
module's `acquire()`/`produce()` run on different threads. A single per-stream
timeline therefore cannot hold module slices without overlaps. So each slice goes
on a **per-(stream, thread) lane** that hangs under the stream:

```
process "cmsRun"
  └─ edm::stream <sid>                  (one per stream; events are serialized here)
        ├─ "Event" slice + run/lumi/event counters
        └─ thread <n>                   (lanes: one per thread that worked on the stream)
              └─ module / acquire / eventsetup / source / cleanup slices
                 (+ alloc/free instants and CMS_PERFETTO_FUNC slices nested under them)
```

- **Module / acquire / EventSetup / source / cleanup** slices are emitted on a lane
  keyed by `(stream, executing thread)`, parented to the stream track. Because every
  lane is fed by exactly one thread, its begin/end events arrive in order and nest
  correctly; modules of one stream running concurrently on different threads simply
  show up as parallel lanes under that stream. The primary thread carries most of a
  stream's work; brief lanes are tasks TBB ran on other threads (work-stealing).
- The **per-stream `Event` track** itself holds the serialized event lifetime
  (`preEvent`..`postClearEvent`) plus the run/lumi/event counters.

The service also publishes a thread-local `cms::perfetto::ModuleContext` around
every module call, so the caching-allocator monitor and the GPU/CUPTI layer can
attribute their work to the responsible module.

## Usage

With `cmsDriver.py`:

```bash
cmsDriver.py step3 -s RAW2DIGI,RECO ... \
  --customise PerfTools/Perfetto/customisePerfetto.customise
```

Or directly in a config:

```python
from PerfTools.Perfetto.customisePerfetto import customisePerfetto
customisePerfetto(process, fileName="reco.pftrace")
```

Or the bare service:

```python
process.add_(cms.Service("PerfettoTraceService",
                         fileName=cms.untracked.string("reco.pftrace")))
```

### Parameters (all untracked)

| parameter        | default          | meaning                                                        |
|------------------|------------------|----------------------------------------------------------------|
| `enabled`        | `True`           | master switch                                                  |
| `fileName`       | `cmsrun.pftrace` | output trace file                                              |
| `bufferSizeKB`   | `262144`         | in-process trace buffer size (KB)                              |
| `maxEvents`      | `200`            | stop opening new event slices after N events (`0` = unlimited) |
| `traceFunctions` | `False`          | enable tier-B per-function slices                              |
| `traceAllocations` | `False`        | trace Alpaka caching-allocator alloc/free + device-memory counters |
| `traceGpuKernels` | `False`         | trace CUDA kernels (real device timing + registers/occupancy) via CUPTI |
| `tracePower`     | `False`          | sample CPU (RAPL) + GPU (NVML) power as counter tracks |
| `powerPeriodMs`  | `1000`           | power sampling period in ms (when `tracePower`) |
| `traceModules`   | `[]`             | if non-empty, only trace these module labels (focused, low overhead) |

A global **`Throughput (events/s)`** counter is always emitted -- a sliding-window
event rate (over the last 16 completed events) that shows the job ramping up to
steady state. With `tracePower=True`, a background thread samples **`CPU pkg<n>
power (W)`** (Intel RAPL, `/sys/class/powercap`) and **`GPU<d> power (W)`** (NVML,
loaded via `dlopen`) every `powerPeriodMs` (default 1000 ms -- NVML power queries
are not free, so polling too fast can perturb the GPU). Both are no-ops where the
source is unavailable.

### Per-function (tier-B) tracing

Annotate hot code with `CMS_PERFETTO_FUNC()` (uses `__func__`) or
`CMS_PERFETTO_SCOPE("name")`. The slices are emitted on the calling thread's
track, nesting under the enclosing module slice, while tracing is active;
otherwise they are no-ops.

```cpp
#include "PerfTools/Perfetto/interface/CMSSWPerfettoTrace.h"

void MyProducer::produce(edm::Event& e, edm::EventSetup const&) {
  CMS_PERFETTO_FUNC();
  ...
}
```

### Caching-allocator tracing (`traceAllocations=True`)

The Alpaka `CachingAllocator` calls an optional `cms::alpakatools::CachingAllocatorMonitor`
(a process-wide hook, free of any perfetto dependency) on every alloc/free. The
service registers `PerfettoAllocatorMonitor`, which emits each transaction as an
INSTANT on the calling thread's track — so it sits under the module slice that
triggered it and is annotated with that module (from `ModuleContext`), the byte
size, cache-hit/miss, device and queue — plus per-device `live`/`cached`/`requested`
byte counters for a device-memory-pressure timeline.

Because a freed block may still be in use by asynchronous device work, the allocator
only re-hands it once a recorded event completes (possibly to another thread/stream):
this asynchronous reuse is visible as a later `alloc` with `cache_hit=true` on a
different thread, against the same device/queue.

### GPU kernel tracing (`traceGpuKernels=True`)

Uses [CUPTI](https://docs.nvidia.com/cupti/) to stream CUDA kernel activity into
the session. CUPTI is a driver-level profiler, so it captures kernels from the
already-built Alpaka/CUDA plugins without rebuilding them, at low overhead and
without serializing kernels. Each kernel is a slice on a per-(device, CUDA stream)
track, placed at its **device-side** start/end — i.e. the *real* GPU execution
time, not the host enqueue/wait — annotated with `registers_per_thread`,
`static_smem_B`, `dynamic_smem_B`, `local_per_thread_B` / `local_total_B` (spills,
per thread and for the whole launch), `grid`, `block`,
an estimated `occupancy_est`, the full kernel name, and the CUPTI `correlation_id`
that links it back to the host module that launched it. GPU timestamps are
converted to `CLOCK_BOOTTIME` so they line up with the host timeline.

The compile-time counterpart is `scripts/perfettoKernelResources.py`, which runs
`cuobjdump --dump-resource-usage` on the built `*PortableCudaAsync.so` libraries to
print the per-kernel registers / shared / stack / spill / constant usage (the
ptxas numbers), e.g.:

```bash
perfettoKernelResources.py --filter Phase2 \
  $CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH/pluginRecoLocalTrackerSiPixelClusterizerPluginsPortableCudaAsync.so
```

## Threading model and nested parallelism

Slices are emitted on the **thread** that does the work, because within one stream
and one event CMSSW runs independent modules concurrently on different threads (and
an ExternalWork module's `acquire()`/`produce()` run on different threads). A few
consequences worth knowing when reading a trace:

- A module's slice spans its `produce()` **wall-clock on that thread**, including
  any blocking `tbb::parallel_for` it calls. The parallel iterations that run on
  *other* threads show up on *those* threads' tracks, not nested under the module —
  so you can see the fan-out, but it is not visually grouped under the module.
- **`tbb::parallel_for` inside `produce()`**: while the calling thread is blocked in
  the `parallel_for`, TBB may run an unrelated module's task on it (work-stealing),
  which then appears *nested inside* your module's slice on that thread. This is a
  faithful picture of what the thread did, but it means a slice's duration can
  include stolen, unrelated work. The thread-local "current module" used for
  allocator/GPU attribution is a **stack**, so it is restored correctly when the
  stolen work returns.
- The helper threads running the `parallel_for` body do **not** automatically carry
  the module context, so allocations / `CMS_PERFETTO_FUNC` slices made there are not
  attributed to the module. Wrap the body to propagate it:

  ```cpp
  #include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"
  tbb::parallel_for(range, cms::perfetto::withModuleContext([&](auto const& r) {
    CMS_PERFETTO_FUNC();           // now attributed to the enclosing module
    ...
  }));
  ```

## A profiling guide for reconstruction developers

Goal: understand and speed up *one* algorithm (a module and the GPU work it drives).

1. **Focus the trace on your module** (much lower overhead, far less noise):

   ```bash
   cmsRun step3.py   # add via --customise_commands or customisePerfetto(...)
   ```
   ```python
   from PerfTools.Perfetto.customisePerfetto import customisePerfetto
   customisePerfetto(process,
                     fileName="myalgo.pftrace",
                     traceModules=["myProducerAlpaka"],   # only this module
                     traceGpuKernels=True,                # its CUDA kernels
                     traceAllocations=True)               # its device memory
   ```

2. **Open `myalgo.pftrace` at <https://ui.perfetto.dev>** and read it top-down:
   - **Per-thread tracks** — find your module's slice. Its width is the host-side
     time (enqueue for an async GPU module; the actual compute for a CPU module).
     If you added `CMS_PERFETTO_FUNC()`/`CMS_PERFETTO_SCOPE()`, the sub-steps nest
     underneath.
   - **`GPU<d> stream <s>` tracks** — the real kernel execution. Click a kernel to
     see `registers_per_thread`, `occupancy_est`, `grid`/`block`, shared/local
     memory. Use the `correlation_id` to match a kernel to the host launch.
   - **`Event` tracks** — per-stream event boundaries + run/lumi/event counters.
   - **`dev<d> live/cached/requested (B)` counters** — the device-memory timeline.

3. **What to look for:**
   - *Kernel time vs host time*: an async GPU module's host slice is tiny while the
     real cost is on the GPU track — optimize the kernel, not the host call.
   - *Occupancy*: a low `occupancy_est` driven by a high `registers_per_thread` (or
     large shared memory) means the kernel is occupancy-limited; cross-check the
     static numbers with `scripts/perfettoKernelResources.py` (it also shows
     register *spills*, `local_spill > 0`).
   - *Memory churn*: `alloc` events with `cache_hit=false` are real `cudaMalloc`s
     (expensive); a sawtooth in the `live` counter means repeated alloc/free that
     the cache is not absorbing — consider reusing buffers or sizing the cache. A
     low cache-hit rate early in a job is normal (cold cache); it should rise as
     blocks are reused.
   - *Allocation overhead*: the gap between the `live` and `requested` counters is
     the power-of-two bin rounding (often tens of %); a few oversized buffers from
     one module usually dominate the peak — group `alloc` events by `module` to
     find them.
   - *Serialization*: the caching allocator takes one global lock per device; if
     `alloc`/`free` instants from many threads line up back-to-back, that lock may
     be a contention point.
   - *Gaps*: an `acquire` slice on one thread, a long gap (GPU + the `edm async pool`
     wait), then `produce` on another thread is the normal ExternalWork pattern; the
     gap is the GPU doing work, visible on the GPU track.

## Overhead

The cost is opt-in and proportional to what you enable:

- Disabled categories cost a predicated load; the `enabled`/`IsEnabled` guard makes
  every callback an early return.
- `traceModules=[...]` is the main overhead lever: only selected modules emit
  slices, so a focused run on one algorithm avoids the hundreds of per-module
  events of a full event. The module-context stack push/pop is allocation-free.
- `traceAllocations`: when off, the allocator hook is a single relaxed atomic load
  per alloc/free; when on, one instant + a few counters per transaction.
- `traceGpuKernels`: CUPTI activity tracing is asynchronous and does not serialize
  kernels; it is the cheapest of the three relative to the information it adds.
- `maxEvents` caps the trace size (and thus buffer pressure) regardless of job length.
