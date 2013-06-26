
import FWCore.ParameterSet.Config as cms

def customise(process):

#    process.dtDataIntegrityUnpacker.inputLabel = cms.untracked.InputTag('rawDataCollector')

#    process.DQMOffline_SecondStep.remove(process.hcalOfflineDQMClient)

    process.options = cms.untracked.PSet(
       wantSummary = cms.untracked.bool(True) 
    )

#    process.load("FWCore.Modules.printContent_cfi")
#    process.myPath1 = cms.Path( process.printContent )

#    process.output.outputCommands.append('drop *')
#    process.output.outputCommands.append('keep *_MEtoEDMConverter_*_*')

#    process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#     ignoreTotal=cms.untracked.int32(1),
#     oncePerEventMode=cms.untracked.bool(False)
#    )

    process.load("DQMServices.Components.DQMStoreStats_cfi")
    process.stats = cms.EndPath(process.dqmStoreStats)

    process.schedule.append(process.stats)
    if hasattr(process, 'dqmSaver'):
      process.dqmSaver.saveByLumiSection = cms.untracked.int32(1)
    
    return(process)

