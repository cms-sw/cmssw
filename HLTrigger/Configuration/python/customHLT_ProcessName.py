import FWCore.ParameterSet.Config as cms

def customise(process):

    process.options.wantSummary = cms.untracked.bool(True)
    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('HLTrigReport')

# re-running HLT requires new process name!

    process.setName_("HLT2")
    process.hltTrigReport.HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT2' )

    return(process)
