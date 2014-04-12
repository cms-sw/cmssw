
import FWCore.ParameterSet.Config as cms

def customise(process):

#    process.dtDataIntegrityUnpacker.inputLabel = cms.untracked.InputTag('rawDataCollector')

#    process.DQMOfflineCosmics.remove(process.hcalOfflineDQMSource)

#    process.load("FWCore.Modules.printContent_cfi")
#    process.myPath1 = cms.Path( process.printContent )

    process.options = cms.untracked.PSet(
      wantSummary = cms.untracked.bool(True) 
    )

    #using the DQMROOT means that the reco output will be empty
    process.DQMoutput.outputCommands.append('drop *')
    process.DQMoutput.outputCommands.append('keep *_MEtoEDMConverter_*_*')

#    process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#     ignoreTotal=cms.untracked.int32(1),
#     oncePerEventMode=cms.untracked.bool(False)
#    )

    # Do not activate by default the logging of where each histogram is booked.
    process.DQMStore.verbose = cms.untracked.int32(2)
    process.load("DQMServices.Components.DQMStoreStats_cfi")
    process.stats = cms.Path(process.dqmStoreStats)
    process.schedule.insert(-2,process.stats)

    return(process)

