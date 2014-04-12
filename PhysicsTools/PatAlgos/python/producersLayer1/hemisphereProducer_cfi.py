import FWCore.ParameterSet.Config as cms

patHemispheres = cms.EDProducer("PATHemisphereProducer",
    patElectrons = cms.InputTag("cleanLayer1Electrons"),
    patJets      = cms.InputTag("cleanLayer1Jets"),
    patMuons     = cms.InputTag("cleanLayer1Muons"),
    patPhotons   = cms.InputTag("cleanLayer1Photons"),
    patTaus      = cms.InputTag("cleanLayer1Taus"),
    patMets      = cms.InputTag("layer1METs"),

    minJetEt = cms.double(30),
    minMuonEt = cms.double(7),
    minElectronEt = cms.double(7),
    minTauEt = cms.double(1000000),
    minPhotonEt = cms.double(200000),
    maxJetEta = cms.double(5),
    maxMuonEta = cms.double(5),
    maxElectronEta = cms.double(5),
    maxTauEta = cms.double(-1),
    maxPhotonEta = cms.double(5),

    seedMethod        = cms.int32(3),
    combinationMethod = cms.int32(3),
)
