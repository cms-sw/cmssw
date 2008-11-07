import FWCore.ParameterSet.Config as cms

def customise(process):

    process.source.inputCommands = cms.untracked.vstring('drop *',
                                                         'keep *_g4SimHits_*_*',
                                                         'keep *_randomEngineStateProducer_*_*',
                                                         'keep edmHepMCProduct_source_*_*',
                                                         'keep *_genEventWeight_*_*',
                                                         'keep *_genEventScale_*_*',
                                                         'keep *_genEventPdfInfo_*_*',
                                                         'keep edmHepMCProduct_source_*_*',
                                                         'keep edmGenInfoProduct_source_*_*',
                                                         'keep *_genEventProcID_*_*',
                                                         'keep *_genEventRunInfo_*_*',
                                                         'keep edmAlpgenInfoProduct_source_*_*',
                                                         'keep edmTriggerResults_*_*_*',
                                                         'keep triggerTriggerEvent_*_*_*',
                                                         'keep *_hltGtDigis_*_*',
                                                         'keep *_hltGctDigis_*_*',
                                                         'drop triggerTriggerEvent_hltTriggerSummaryAOD_*_*',
                                                         'drop L1*_hltGtDigis_*_*',
                                                         'drop L1*_hltGctDigis_*_*')

    if hasattr(process,"RandomNumberGeneratorService"):
        del process.RandomNumberGeneratorService.theSource
    else:    
        process.load("IOMC/RandomEngine/IOMC_cff")
        del process.RandomNumberGeneratorService.theSource

    process.RandomNumberGeneratorService.restoreStateLabel = cms.untracked.string('randomEngineStateProducer')
    process.mix.playback = cms.untracked.bool(True)
    
    # Output definition for RAW
    process.outputRaw = cms.OutputModule("PoolOutputModule",
       outputCommands = process.RAWSIMEventContent.outputCommands,
       fileName = cms.untracked.string('New_RAWSIM.root'),
       dataset = cms.untracked.PSet(
           dataTier = cms.untracked.string(''),
           filterName = cms.untracked.string('')
       )
    )

    process.out_step_raw = cms.EndPath(process.outputRaw)
    process.schedule.append(process.out_step_raw)
    
    return(process)
