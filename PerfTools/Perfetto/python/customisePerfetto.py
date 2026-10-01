# Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
import FWCore.ParameterSet.Config as cms

_TYPES = {
    "enabled": cms.untracked.bool,
    "fileName": cms.untracked.string,
    "bufferSizeKB": cms.untracked.uint32,
    "shmemSizeKB": cms.untracked.uint32,
    "maxEvents": cms.untracked.uint32,
    "traceFunctions": cms.untracked.bool,
    "traceAllocations": cms.untracked.bool,
    "traceGpuKernels": cms.untracked.bool,
    "tracePower": cms.untracked.bool,
    "powerPeriodMs": cms.untracked.uint32,
    "traceModules": cms.untracked.vstring,
}


def customisePerfetto(process, **params):
    """Add the PerfettoTraceService to process.

    Keyword arguments are the service parameters (see the PerfTools/Perfetto
    README); those not given keep the service defaults, e.g.
      customisePerfetto(process, fileName="reco.pftrace", traceModules=["myProducer"])
    """
    unknown = set(params) - set(_TYPES)
    if unknown:
        raise TypeError("customisePerfetto: unknown parameter(s) " + ", ".join(sorted(unknown)))
    process.add_(cms.Service("PerfettoTraceService", **{k: _TYPES[k](v) for k, v in params.items()}))
    return process


def customise(process):
    """Entry point for `cmsDriver.py --customise PerfTools/Perfetto/customisePerfetto.customise`."""
    return customisePerfetto(process)
