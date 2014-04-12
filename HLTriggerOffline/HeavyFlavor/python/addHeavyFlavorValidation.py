import FWCore.ParameterSet.Config as cms

def customise(process):
    process.load("HLTriggerOffline.HeavyFlavor.heavyFlavorValidationSequence_cff")
    process.heavyFlavorValidation_step = cms.Path(process.heavyFlavorValidationSequence)
    process.schedule.insert( process.schedule.index(process.endjob_step), process.heavyFlavorValidation_step )
    process.output.outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_MEtoEDMConverter_*_*'
    )
    process.output.fileName = cms.untracked.string('/tmp/heavyFlavorValidation.root')
    return process

