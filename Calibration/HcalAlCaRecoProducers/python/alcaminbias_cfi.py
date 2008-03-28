import FWCore.ParameterSet.Config as cms

# producer for alcaminbisas (HCAL minimum bias)
MinProd = cms.EDProducer("AlCaEcalHcalReadoutsProducer",
    hbheInput = cms.InputTag("hbhereco"),
    hfInput = cms.InputTag("hfreco"),
    hoInput = cms.InputTag("horeco")
)


