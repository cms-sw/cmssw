import FWCore.ParameterSet.Config as cms

# Generic jetTag configuration
cTagCorrelationAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        Discr1Start = cms.double(-1.011),
        Discr2Start = cms.double(-1.011),
        Discr1End = cms.double(1.011),
        Discr2End = cms.double(1.011),
        nBinEffPur = cms.int32(200),
        startEffPur = cms.double(0.005),
        endEffPur = cms.double(1.005),
        CreateProfile = cms.bool(False),
        fixedEff = cms.vdouble(0.2,0.3,0.4,0.5)
    )
)


