import FWCore.ParameterSet.Config as cms

hltEgammaCandidatesL1Seeded = cms.EDProducer("EgammaHLTRecoEcalCandidateProducers",
    recoEcalCandidateCollection = cms.string(''),
    scHybridBarrelProducer = cms.InputTag("hltParticleFlowSuperClusterECALL1Seeded","hltParticleFlowSuperClusterECALBarrel"),
    scIslandEndcapProducer = cms.InputTag("particleFlowSuperClusterHGCalFromTICLL1Seeded")
)
