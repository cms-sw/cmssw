import FWCore.ParameterSet.Config as cms

# MkFitEventOfHits options
hltMkFitEventOfHits = cms.EDProducer("MkFitEventOfHitsProducer",
    beamSpot = cms.InputTag("offlineBeamSpot"),
    mightGet = cms.optional.untracked.vstring,
    pixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    stripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    usePixelQualityDB = cms.bool(True),
    useStripStripQualityDB = cms.bool(False)
)
