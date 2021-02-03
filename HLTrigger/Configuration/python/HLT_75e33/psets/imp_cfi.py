import FWCore.ParameterSet.Config as cms

imp = cms.PSet(
    importerName = cms.string('GenericClusterImporter'),
    source = cms.InputTag("particleFlowClusterPS")
)