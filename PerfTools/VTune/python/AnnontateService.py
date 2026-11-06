import FWCore.ParameterSet.Config as cms

def customiseVTuneAnnotateService(process):
    from PerfTools.VTune.modules import VTuneAnnotateService
    process.add_(VTuneAnnotateService(allModules = True))
    return(process)
