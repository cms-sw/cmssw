import FWCore.ParameterSet.Config as cms

def customiseVTuneFilterService(process):
    from PerfTools.VTune.modules import VTuneFilterService
    process.add_(VTuneFilterService(allModules = True))
    return (process)
