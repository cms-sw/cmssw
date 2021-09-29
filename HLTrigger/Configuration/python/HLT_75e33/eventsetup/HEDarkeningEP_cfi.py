import FWCore.ParameterSet.Config as cms

HEDarkeningEP = cms.ESSource("HBHEDarkeningEP",
    appendToDataLabel = cms.string('HE'),
    dosemaps = cms.VPSet(
        cms.PSet(
            energy = cms.int32(8),
            file = cms.FileInPath('CalibCalorimetry/HcalPlugins/data/dosemapHE_4TeV.txt')
        ),
        cms.PSet(
            energy = cms.int32(14),
            file = cms.FileInPath('CalibCalorimetry/HcalPlugins/data/dosemapHE_7TeV.txt')
        )
    ),
    drdA = cms.double(2.7383),
    drdB = cms.double(0.37471),
    ieta_shift = cms.int32(16),
    years = cms.VPSet(
        cms.PSet(
            energy = cms.int32(8),
            intlumi = cms.double(5.6),
            lumirate = cms.double(0.005),
            year = cms.string('2011')
        ),
        cms.PSet(
            energy = cms.int32(8),
            intlumi = cms.double(23.3),
            lumirate = cms.double(0.013),
            year = cms.string('2012')
        ),
        cms.PSet(
            energy = cms.int32(14),
            intlumi = cms.double(4.1),
            lumirate = cms.double(0.009),
            year = cms.string('2015')
        ),
        cms.PSet(
            energy = cms.int32(14),
            intlumi = cms.double(41.0),
            lumirate = cms.double(0.026),
            year = cms.string('2016')
        ),
        cms.PSet(
            energy = cms.int32(14),
            intlumi = cms.double(45.0),
            lumirate = cms.double(0.043),
            year = cms.string('2017')
        ),
        cms.PSet(
            energy = cms.int32(14),
            intlumi = cms.double(45.0),
            lumirate = cms.double(0.043),
            year = cms.string('2018')
        ),
        cms.PSet(
            energy = cms.int32(14),
            intlumi = cms.double(45.0),
            lumirate = cms.double(0.05),
            year = cms.string('2021')
        ),
        cms.PSet(
            energy = cms.int32(14),
            intlumi = cms.double(45.0),
            lumirate = cms.double(0.05),
            year = cms.string('2022')
        ),
        cms.PSet(
            energy = cms.int32(14),
            intlumi = cms.double(50.0),
            lumirate = cms.double(0.05),
            year = cms.string('2023')
        ),
        cms.PSet(
            energy = cms.int32(14),
            intlumi = cms.double(2700),
            lumirate = cms.double(0.15),
            year = cms.string('2038')
        )
    )
)
