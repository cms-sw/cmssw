import FWCore.ParameterSet.Config as cms

cosmictrackingParticleSelector = cms.EDFilter("CosmicTrackingParticleSelector",
    src = cms.InputTag("mix","MergedTrackTruth"),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    tip = cms.double(100),
    minRapidity = cms.double(-2.4),
    lip = cms.double(100.0),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.4),
    minHit = cms.int32(0)
   )



