import FWCore.ParameterSet.Config as cms

def customise(process):

    process.schedule.remove(process.L1simulation_step)

    process.hltTrigReport.HLTriggerResults = cms.InputTag( 'TriggerResults','',process.name_() )

    process.options.wantSummary = cms.untracked.bool(True)
    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('L1GtTrigReport')
    process.MessageLogger.categories.append('HLTrigReport')

    return(process)
