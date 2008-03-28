import FWCore.ParameterSet.Config as cms

# producer for alcadijets (HCAL di-jets)
DiJProd = cms.EDProducer("AlCaDiJetsProducer",
    hbheInput = cms.InputTag("hbhereco"),
    hfInput = cms.InputTag("hfreco"),
    hoInput = cms.InputTag("horeco"),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    #                VInputTag srcCalo = {iterativeCone5CaloJets, midPointCone5CaloJets, midPointCone7CaloJets,ktCaloJets}
    srcCalo = cms.VInputTag(cms.InputTag("iterativeCone7CaloJets")),
    inputTrackLabel = cms.untracked.string('generalTracks')
)


