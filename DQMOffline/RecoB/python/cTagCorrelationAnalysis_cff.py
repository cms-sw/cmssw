import FWCore.ParameterSet.Config as cms

# Generic jetTag configuration
cTagCorrelationAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        Discr1Start = cms.double(-1.011),
        Discr2Start = cms.double(-1.011),
        Discr1End = cms.double(1.011),
        Discr2End = cms.double(1.011),
        CreateProfile = cms.bool(True)
    )
)


