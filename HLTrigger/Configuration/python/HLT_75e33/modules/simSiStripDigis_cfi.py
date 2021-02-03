import FWCore.ParameterSet.Config as cms

simSiStripDigis = cms.EDAlias(
    mix = cms.VPSet(
        cms.PSet(
            type = cms.string('SiStripDigiedmDetSetVector')
        ),
        cms.PSet(
            type = cms.string('SiStripRawDigiedmDetSetVector')
        ),
        cms.PSet(
            type = cms.string('StripDigiSimLinkedmDetSetVector')
        )
    )
)