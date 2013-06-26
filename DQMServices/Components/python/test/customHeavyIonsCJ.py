
import FWCore.ParameterSet.Config as cms

def customise(process):

    process.source.fileNames = [ '/store/hidata/HIRun2010/HIAllPhysics/RAW/v1/000/150/887/FC4C9779-03EE-DF11-AAC6-001617C3B778.root']

    # Test CJons new intermediate DQM format!
    process.schedule.remove(process.endjob_step)
    process.out = cms.OutputModule("DQMRootOutputModule",
                                   fileName = cms.untracked.string("testCJones_RunLumi.root"))
    process.endjobCJ_step = cms.EndPath(process.out)
    process.schedule.insert(-2,process.endjobCJ_step)

    return(process)

