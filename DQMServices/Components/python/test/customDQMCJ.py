
import FWCore.ParameterSet.Config as cms

def customise(process):

    process.options = cms.untracked.PSet(
      wantSummary = cms.untracked.bool(True) 
    )

    process.DQMoutput.outputCommands.append('drop *')
    process.DQMoutput.outputCommands.append('keep *_MEtoEDMConverter_*_*')

    process.load("DQMServices.Components.DQMStoreStats_cfi")
    process.stats = cms.Path(process.dqmStoreStats)
    process.schedule.insert(-2,process.stats)

    # Test CJons new intermediate DQM format!
    process.schedule.remove(process.endjob_step)
    process.out = cms.OutputModule("DQMRootOutputModule",
                                   fileName = cms.untracked.string("testCJones_RunLumi.root"))
    process.endjobCJ_step = cms.EndPath(process.out)
    process.schedule.insert(-2,process.endjobCJ_step)

    return(process)

