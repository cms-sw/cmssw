import FWCore.ParameterSet.Config as cms

def customise(process):

    process.options.wantSummary = cms.untracked.bool(True)

    process.load('L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi')
    process.l1GtTrigReport.L1GtRecordInputTag = cms.InputTag( "simGtDigis" )

    process.L1AnalyzerEndpath = cms.EndPath( process.l1GtTrigReport )
    process.schedule.append(process.L1AnalyzerEndpath)

    process.MessageLogger.categories.append('L1GtTrigReport')

    return(process)
