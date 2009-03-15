import FWCore.ParameterSet.Config as cms

def customise(process):
    
    process.l1GtTrigReport = cms.EDAnalyzer( "L1GtTrigReport",
        UseL1GlobalTriggerRecord = cms.bool( False ),
        L1GtRecordInputTag = cms.InputTag( "simGtDigis" ),
        PrintOutput = cms.untracked.int32( 0 )
    )
    process.L1AnalyzerEndpath = cms.EndPath (process.l1GtTrigReport)
    process.schedule.append(process.L1AnalyzerEndpath)
    
    process.options.wantSummary = cms.untracked.bool(True)

    process.MessageLogger.categories.append('L1GtTrigReport')

    return(process)
