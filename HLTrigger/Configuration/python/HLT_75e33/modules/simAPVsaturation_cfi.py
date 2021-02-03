import FWCore.ParameterSet.Config as cms

simAPVsaturation = cms.EDAlias(
    mix = cms.VPSet(cms.PSet(
        type = cms.string('bool')
    ))
)