import FWCore.ParameterSet.Config as cms

simPFProducer = cms.EDProducer(
    'SimPFProducer',
    superClusterThreshold = cms.double(4.0),
    trackSrc = cms.InputTag('generalTracks'),
    gsfTrackSrc = cms.InputTag('electronGsfTracks'),
    trackingParticleSrc = cms.InputTag('mix:MergedTrackTruth'),
    simClusterTruthSrc = cms.InputTag('mix:MergedCaloTruth'),
    caloParticlesSrc = cms.InputTag('mix:MergedCaloTruth'),
    simClustersSrc = cms.InputTag('particleFlowClusterHGCal'),
    associators = cms.VInputTag(cms.InputTag('quickTrackAssociatorByHits') )
    )
