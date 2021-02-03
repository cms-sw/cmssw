import FWCore.ParameterSet.Config as cms

simEcalUnsuppressedDigis = cms.EDAlias(
    mix = cms.VPSet(
        cms.PSet(
            type = cms.string('EBDigiCollection')
        ),
        cms.PSet(
            type = cms.string('EEDigiCollection')
        ),
        cms.PSet(
            type = cms.string('ESDigiCollection')
        )
    )
)