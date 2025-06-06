import FWCore.ParameterSet.Config as cms
def customise(process):
    process.AllocMonitor = cms.Service("ModuleAllocMonitor",
                                   fileName=cms.untracked.string("moduleAllocMonitor.log")
                                   )
    return(process)
