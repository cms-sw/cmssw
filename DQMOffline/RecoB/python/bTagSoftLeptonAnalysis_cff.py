import FWCore.ParameterSet.Config as cms

# SoftLepton jetTag configuration
bTagSoftLeptonAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.1),
        nBinEffPur = cms.int32(200),
        # the constant b-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.05),
        endEffPur = cms.double(0.205),
        discriminatorEnd = cms.double(1.1),
        startEffPur = cms.double(0.005)
    )
)
