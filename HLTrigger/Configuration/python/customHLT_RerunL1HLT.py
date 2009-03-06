import FWCore.ParameterSet.Config as cms

from L1Trigger.Configuration import patchToRerunL1Emulator

def customise(process):

    process.options.wantSummary = cms.untracked.bool(True)
    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('HLTrigReport')

# re-running HLT requires new process name!

    process.setName_("HLT2")
    process.hltTrigReport.HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT2' )

# run trigger primitive generation on unpacked digis, then central L1

    process.load("L1Trigger.Configuration.CaloTriggerPrimitives_cff")
    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
    process.simHcalTriggerPrimitiveDigis.inputLabel = 'hcalDigis'

# patch the process to use 'sim*Digis' from the L1 emulator
# instead of 'hlt*Digis' from the RAW data

    patchToRerunL1Emulator.switchToSimGtDigis( process )

    return(process)
