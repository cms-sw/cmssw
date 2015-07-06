import FWCore.ParameterSet.Config as cms

modifiedTaus = cms.EDProducer(
    "ModifiedTauProducer",
    src = cms.InputTag("slimmedTaus",processName=cms.InputTag.skipCurrentProcess()),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)
