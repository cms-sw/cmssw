import FWCore.ParameterSet.Config as cms

simHGCalUnsuppressedDigis = cms.EDAlias(
    mix = cms.VPSet(
        cms.PSet(
            fromProductInstance = cms.string('HGCDigisEE'),
            toProductInstance = cms.string('EE'),
            type = cms.string('DetIdHGCSampleHGCDataFramesSorted')
        ),
        cms.PSet(
            fromProductInstance = cms.string('HGCDigisHEfront'),
            toProductInstance = cms.string('HEfront'),
            type = cms.string('DetIdHGCSampleHGCDataFramesSorted')
        ),
        cms.PSet(
            fromProductInstance = cms.string('HGCDigisHEback'),
            toProductInstance = cms.string('HEback'),
            type = cms.string('DetIdHGCSampleHGCDataFramesSorted')
        )
    )
)