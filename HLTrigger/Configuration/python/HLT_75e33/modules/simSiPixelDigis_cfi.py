import FWCore.ParameterSet.Config as cms

simSiPixelDigis = cms.EDAlias(
    mix = cms.VPSet(
        cms.PSet(
            type = cms.string('PixelDigiedmDetSetVector')
        ),
        cms.PSet(
            type = cms.string('PixelDigiSimLinkedmDetSetVector')
        )
    )
)