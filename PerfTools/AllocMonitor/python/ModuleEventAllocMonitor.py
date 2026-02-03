import FWCore.ParameterSet.Config as cms
def customise(process):
    process.ModuleEventAllocMonitor = cms.Service("ModuleEventAllocMonitor",
                                   fileName=cms.untracked.string("moduleEventAllocMonitor.log"),
                                   skippedModuleNames=cms.untracked.vstring('mix',),
                                   )
    return(process)
