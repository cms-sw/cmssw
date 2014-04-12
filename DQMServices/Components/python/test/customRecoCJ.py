
import FWCore.ParameterSet.Config as cms

def customise(process):

#    process.dtDataIntegrityUnpacker.inputLabel = cms.untracked.InputTag('rawDataCollector')

#    process.DQMOfflineCosmics.remove(process.hcalOfflineDQMSource)

#    process.load("FWCore.Modules.printContent_cfi")
#    process.myPath1 = cms.Path( process.printContent )

    process.options = cms.untracked.PSet(
      wantSummary = cms.untracked.bool(True) 
    )

    process.RECOoutput.outputCommands.append('drop *')
    process.RECOoutput.outputCommands.append('keep *_MEtoEDMConverter_*_*')

#    process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#     ignoreTotal=cms.untracked.int32(1),
#     oncePerEventMode=cms.untracked.bool(False)
#    )

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

