import FWCore.ParameterSet.Config as cms

from L1Trigger.Configuration import patchToRerunL1Emulator

def customise(process):
    process.setName_('HLT2')

    process.hltL1GtTrigReport = cms.EDAnalyzer( "L1GtTrigReport",
        PrintVerbosity = cms.untracked.int32(0),
        PrintOutput = cms.untracked.int32(2),
        UseL1GlobalTriggerRecord = cms.bool( False ),
        L1GtRecordInputTag = cms.InputTag( "hltGtDigis" )
    )
    process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
        HLTriggerResults = cms.InputTag( 'TriggerResults','',process.name_() )
    )
    process.HLTAnalyzerEndpath = cms.EndPath( process.hltL1GtTrigReport + process.hltTrigReport )
    process.schedule.append(process.HLTAnalyzerEndpath)

    process.options.wantSummary = cms.untracked.bool(True)
    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('L1GtTrigReport')
    process.MessageLogger.categories.append('HLTrigReport')

    # run trigger primitive generation on unpacked digis, then central L1
    process.load("L1Trigger.Configuration.CaloTriggerPrimitives_cff")
    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
    process.simHcalTriggerPrimitiveDigis.inputLabel = 'hcalDigis'

    # patch the process to use 'sim*Digis' from the L1 emulator
    # instead of 'hlt*Digis' from the RAW data
    patchToRerunL1Emulator.switchToSimGtDigis( process )

    return(process)
