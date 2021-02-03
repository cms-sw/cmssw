import FWCore.ParameterSet.Config as cms

simHcalUnsuppressedDigis = cms.EDAlias(
    mix = cms.VPSet(
        cms.PSet(
            type = cms.string('HBHEDataFramesSorted')
        ),
        cms.PSet(
            type = cms.string('HFDataFramesSorted')
        ),
        cms.PSet(
            type = cms.string('HODataFramesSorted')
        ),
        cms.PSet(
            type = cms.string('ZDCDataFramesSorted')
        ),
        cms.PSet(
            type = cms.string('QIE10DataFrameHcalDataFrameContainer')
        ),
        cms.PSet(
            type = cms.string('QIE11DataFrameHcalDataFrameContainer')
        )
    )
)