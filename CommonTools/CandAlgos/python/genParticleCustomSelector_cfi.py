import FWCore.ParameterSet.Config as cms

genParticleCustomSelector = cms.EDFilter("GenParticleCustomSelector",
    src = cms.InputTag("genParticles"),
    chargedOnly = cms.bool(True),
    status = cms.int32(1),
    pdgId = cms.vint32(),
    tip = cms.double(3.5),
    minRapidity = cms.double(-2.4),
    lip = cms.double(30.0),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.4),
)



