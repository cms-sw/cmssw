import FWCore.ParameterSet.Config as cms

trackingParticleRefSelector = cms.EDFilter("TrackingParticleRefSelector",
    src = cms.InputTag("mix","MergedTrackTruth"),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    tip = cms.double(3.5),
    signalOnly = cms.bool(True),
    intimeOnly = cms.bool(False),
    stableOnly = cms.bool(False),
    minRapidity = cms.double(-2.4),
    lip = cms.double(30.0),
    ptMin = cms.double(0.9),
    ptMax = cms.double(1e100),
    maxRapidity = cms.double(2.4),
    minHit = cms.int32(0),
    minPhi = cms.double(-3.2),
    maxPhi = cms.double(3.2),
    invertRapidityCut = cms.bool(False)
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(trackingParticleRefSelector, src = "mixData:MergedTrackTruth")
