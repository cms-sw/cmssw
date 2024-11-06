import FWCore.ParameterSet.Config as cms

hltEgammaCandidatesUnseeded = cms.EDProducer("EgammaHLTRecoEcalCandidateProducers",
    recoEcalCandidateCollection = cms.string(''),
    scHybridBarrelProducer = cms.InputTag("hltParticleFlowSuperClusterECALUnseeded","particleFlowSuperClusterECALBarrel"),
    scIslandEndcapProducer = cms.InputTag("hltParticleFlowSuperClusterHGCalFromTICLUnseeded")
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(hltEgammaCandidatesUnseeded, scIslandEndcapProducer = cms.InputTag("hltTiclEGammaSuperClusterProducerUnseeded"))
