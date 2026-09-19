# Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
import FWCore.ParameterSet.Config as cms


def customisePerfetto(process,
                      fileName="cmsrun.pftrace",
                      bufferSizeKB=256 * 1024,
                      maxEvents=200,
                      traceFunctions=False,
                      traceAllocations=False,
                      traceGpuKernels=False,
                      traceModules=None):
    """Add the PerfettoTraceService to process.

    The service writes an in-process Perfetto trace (open it at https://ui.perfetto.dev).
    Module / acquire / EventSetup slices land on per-thread tracks (so concurrent
    work nests correctly); each stream gets an "Event" track with run/lumi/event
    counters.

    Arguments:
      fileName         output .pftrace file
      bufferSizeKB     in-process trace buffer size (KB)
      maxEvents        stop opening new event slices after this many events (0 = unlimited)
      traceFunctions   enable tier-B per-function slices (CMS_PERFETTO_FUNC/SCOPE)
      traceAllocations trace the Alpaka caching allocator (alloc/free + device-memory counters)
      traceGpuKernels  trace CUDA kernels via CUPTI (real device timing + registers/occupancy)
      traceModules     if a non-empty list, only trace these module labels (focused, low overhead)
    """
    process.add_(cms.Service("PerfettoTraceService",
                             fileName=cms.untracked.string(fileName),
                             bufferSizeKB=cms.untracked.uint32(bufferSizeKB),
                             maxEvents=cms.untracked.uint32(maxEvents),
                             traceFunctions=cms.untracked.bool(traceFunctions),
                             traceAllocations=cms.untracked.bool(traceAllocations),
                             traceGpuKernels=cms.untracked.bool(traceGpuKernels),
                             traceModules=cms.untracked.vstring(traceModules or [])))
    return process


def customise(process):
    """Default entry point for `cmsDriver.py --customise PerfTools/Perfetto/customisePerfetto.customise`."""
    return customisePerfetto(process)
