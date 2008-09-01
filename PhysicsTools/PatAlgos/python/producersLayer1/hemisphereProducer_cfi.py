import FWCore.ParameterSet.Config as cms

selectedLayer1Hemispheres = cms.EDProducer("PATHemisphereProducer",
    patElectrons = cms.InputTag("selectedLayer1Electrons"),
    patJets      = cms.InputTag("selectedLayer1Jets"),
    patMets      = cms.InputTag("selectedLayer1METs"),
    patMuons     = cms.InputTag("selectedLayer1Muons"),
    patPhotons   = cms.InputTag("selectedLayer1Photons"),
    patTaus      = cms.InputTag("selectedLayer1Taus"),

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


