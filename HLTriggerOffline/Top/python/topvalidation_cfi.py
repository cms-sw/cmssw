import FWCore.ParameterSet.Config as cms

HLTTopVal = cms.EDAnalyzer("TopValidation",
    OutputMEsInRootFile = cms.bool(False),
    TriggerResultsCollection = cms.InputTag("TriggerResults","","HLT"),
    OutputFileName = cms.string(''),
    DQMFolder = cms.untracked.string("HLT/Top")
 )
