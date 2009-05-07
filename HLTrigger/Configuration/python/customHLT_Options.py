import FWCore.ParameterSet.Config as cms

def customise(process):

    process.schedule.remove(process.L1simulation_step)

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

    return(process)
