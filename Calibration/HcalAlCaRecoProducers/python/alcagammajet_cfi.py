import FWCore.ParameterSet.Config as cms

# producer for alcadijets (HCAL gamma-jet)
GammaJetProd = cms.EDProducer("AlCaGammaJetProducer",
    hbheInput = cms.InputTag("hbhereco"),
    correctedIslandBarrelSuperClusterCollection = cms.string(''),
    correctedIslandEndcapSuperClusterCollection = cms.string(''),
    hfInput = cms.InputTag("hfreco"),
    hoInput = cms.InputTag("horeco"),
    correctedIslandEndcapSuperClusterProducer = cms.string('correctedIslandEndcapSuperClusters'),
    correctedIslandBarrelSuperClusterProducer = cms.string('correctedIslandBarrelSuperClusters'),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    #                VInputTag srcCalo = {iterativeCone5CaloJets, midPointCone5CaloJets, midPointCone7CaloJets,ktCaloJets}
    srcCalo = cms.VInputTag(cms.InputTag("iterativeCone7CaloJets")),
    inputTrackLabel = cms.untracked.string('generalTracks')
)


