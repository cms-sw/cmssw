import FWCore.ParameterSet.Config as cms

# BProbabilty jetTag configuration
bTagBProbabilityAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.08),
        nBinEffPur = cms.int32(200),
        # the constant b-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.5),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(8.08),
        startEffPur = cms.double(0.005)
    )
)


