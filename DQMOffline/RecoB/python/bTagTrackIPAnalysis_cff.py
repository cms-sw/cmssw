import FWCore.ParameterSet.Config as cms

# TrackIP tag info configuration
bTagTrackIPAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        QualityPlots = cms.bool(False),
        endEffPur = cms.double(1.005),
        nBinEffPur = cms.int32(200),
        startEffPur = cms.double(0.005),
        LowerIPSBound = cms.double(-35.0),
        UpperIPSBound = cms.double(35.0),
        LowerIPBound = cms.double(-0.1),
        UpperIPBound = cms.double(0.1),
        LowerIPEBound = cms.double(0.0),
        UpperIPEBound = cms.double(0.04),
        NBinsIPS = cms.int32(100),
        NBinsIP = cms.int32(100),
        NBinsIPE = cms.int32(100),
        MinDecayLength = cms.double(-9999.0),
        MaxDecayLength = cms.double(5.0),
        MinJetDistance = cms.double(0.0),
        MaxJetDistance = cms.double(0.07),
    )
)


