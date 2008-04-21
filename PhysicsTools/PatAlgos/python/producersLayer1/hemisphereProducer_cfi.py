import FWCore.ParameterSet.Config as cms

selectedLayer1Hemispheres = cms.EDProducer("PATHemisphereProducer",
    patJets = cms.InputTag("selectedLayer1Jets"),
    patMuons = cms.InputTag("selectedLayer1Muons"),
    seedMethod = cms.int32(3),
    patElectrons = cms.InputTag("selectedLayer1Electrons"),
    patMets = cms.InputTag("selectedLayer1METs"),
    patTaus = cms.InputTag("selectedLayer1Taus"),
    combinationMethod = cms.int32(3),
    patPhotons = cms.InputTag("selectedLayer1Photons")
)


