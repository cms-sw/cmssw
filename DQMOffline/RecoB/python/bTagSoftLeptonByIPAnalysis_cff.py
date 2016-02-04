import FWCore.ParameterSet.Config as cms

# SoftLepton jetTag configuration
bTagSoftLeptonByIPAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-10.0),
        discriminatorEnd   = cms.double(30.0),
        
        nBinEffPur  = cms.int32(200),
        startEffPur = cms.double(0.005),
        endEffPur   = cms.double(0.205),

        # the constant b-efficiency for the differential plots versus pt and eta
        effBConst   = cms.double(0.05)
    )
)
