import FWCore.ParameterSet.Config as cms


selectedPfJets = cms.EDFilter(
    "GenericPFJetSelector",
    src = cms.InputTag('pfJets'),
    cut = cms.string('')
    )
# foo bar baz
# Oat4QsRJNh8PN
# vtYxTKBIWratK
