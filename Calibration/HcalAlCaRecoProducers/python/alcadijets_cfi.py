import FWCore.ParameterSet.Config as cms

# producer for alcadijets (HCAL di-jets)


DiJProd = cms.EDProducer("AlCaDiJetsProducer",
    jetsInput = cms.InputTag("iterativeCone5CaloJets"),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    hbheInput = cms.InputTag("hbhereco"),
    hoInput = cms.InputTag("horeco"),
    hfInput = cms.InputTag("hfreco")
)

