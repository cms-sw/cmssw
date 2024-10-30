import FWCore.ParameterSet.Config as cms

bTagGenericAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.01),
        nBinEffPur = cms.int32(200),
        # the constant b-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.7),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(1.011),
        startEffPur = cms.double(0.005)
    )
)

cTagGenericAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.01),
        nBinEffPur = cms.int32(200),
        # the constant c-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.7),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(1.011),
        startEffPur = cms.double(0.005)
    )
)

tauTagGenericAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.01),
        nBinEffPur = cms.int32(200),
        # the constant tau-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.7),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(1.011),
        startEffPur = cms.double(0.005)
    )
)

sTagGenericAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.01),
        nBinEffPur = cms.int32(200),
        # the constant s-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.7),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(1.011),
        startEffPur = cms.double(0.005)
    )
)

qgTagGenericAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.01),
        nBinEffPur = cms.int32(200),
        # the constant q-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.7),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(1.011),
        startEffPur = cms.double(0.005)
    )
)

