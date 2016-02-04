import FWCore.ParameterSet.Config as cms

# producer for alcaminbisas (HCAL minimum bias)

MinProd = cms.EDProducer("AlCaEcalHcalReadoutsProducer",
    hbheInput = cms.InputTag("hbherecoMB"),
    hfInput = cms.InputTag("hfrecoMB"),
    hoInput = cms.InputTag("horecoMB")
)


