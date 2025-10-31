import FWCore.ParameterSet.Config as cms

# MkFitSiPixelHits options
hltMkFitSiPixelHits = cms.EDProducer("MkFitSiPixelHitConverter",
        hits = cms.InputTag("hltSiPixelRecHits"),
        clusters = cms.InputTag("hltSiPixelClusters"),
        mightGet = cms.optional.untracked.vstring,
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
)
