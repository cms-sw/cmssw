import FWCore.ParameterSet.Config as cms

trackTimeValueMapProducer = cms.EDProducer(
    'TrackTimeValueMapProducer',
    trackSrc = cms.InputTag('generalTracks'),
    gsfTrackSrc = cms.InputTag('electronGsfTracks'),
    trackingParticleSrc = cms.InputTag('mix:MergedTrackTruth'),
    trackingVertexSrc = cms.InputTag('mix:MergedTrackTruth'),
    tpAssociator = cms.string('quickTrackAssociatorByHits'),
    resolutionModels = cms.VPSet( cms.PSet( modelName = cms.string('ConfigurableFlatResolutionModel'),
                                            resolutionInNs = cms.double(0.020) ) )
    )
