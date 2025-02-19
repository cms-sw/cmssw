import FWCore.ParameterSet.Config as cms

l1RctValidation = cms.EDAnalyzer("L1RCTRelValAnalyzer",
    rctRegionsLabel = cms.InputTag("rctDigis"),
    rctEmCandsLabel = cms.InputTag("rctDigis")
)



