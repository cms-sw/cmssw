import FWCore.ParameterSet.Config as cms

selectedLayer1Hemispheres = cms.EDProducer("PATHemisphereProducer",
    patElectrons = cms.InputTag("selectedLayer1Electrons"),
    patJets      = cms.InputTag("selectedLayer1Jets"),
    patMets      = cms.InputTag("selectedLayer1METs"),
    patMuons     = cms.InputTag("selectedLayer1Muons"),
    patPhotons   = cms.InputTag("selectedLayer1Photons"),
    patTaus      = cms.InputTag("selectedLayer1Taus"),
    seedMethod        = cms.int32(3),
    combinationMethod = cms.int32(3),
)


