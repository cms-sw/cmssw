import FWCore.ParameterSet.Config as cms

# SoftLepton jetTag configuration
bTagSoftLeptonByPtAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.01),
        discriminatorEnd   = cms.double(8.01),

        nBinEffPur  = cms.int32(200),
        startEffPur = cms.double(0.005),
        endEffPur   = cms.double(0.205),

        # the constant b-efficiency for the differential plots versus pt and eta
        effBConst   = cms.double(0.05)
    )
)
