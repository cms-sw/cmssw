import FWCore.ParameterSet.Config as cms

hltEgammaCandidatesUnseeded = cms.EDProducer("EgammaHLTRecoEcalCandidateProducers",
    recoEcalCandidateCollection = cms.string(''),
    scHybridBarrelProducer = cms.InputTag("hltParticleFlowSuperClusterECALUnseeded","particleFlowSuperClusterECALBarrel"),
    scIslandEndcapProducer = cms.InputTag("hltTiclEGammaSuperClusterProducerUnseeded")
)

