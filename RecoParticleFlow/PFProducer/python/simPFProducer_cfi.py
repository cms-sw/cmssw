import FWCore.ParameterSet.Config as cms

simPFProducer = cms.EDProducer(
    'SimPFProducer',
    superClusterThreshold = cms.double(4.0),
    neutralEMThreshold = cms.double(0.250),
    neutralHADThreshold = cms.double(0.250),
    pfRecTrackSrc = cms.InputTag("hgcalTrackCollection:TracksInHGCal"),
    trackSrc = cms.InputTag('generalTracks'),
    gsfTrackSrc = cms.InputTag('electronGsfTracks'),
    muonSrc = cms.InputTag("muons1stStep"),
    trackingParticleSrc = cms.InputTag('mix:MergedTrackTruth'),
    simClusterTruthSrc = cms.InputTag('mix:MergedCaloTruth'),
    caloParticlesSrc = cms.InputTag('mix:MergedCaloTruth'),
    simClustersSrc = cms.InputTag('particleFlowClusterHGCal'),
    associators = cms.VInputTag(cms.InputTag('quickTrackAssociatorByHits') )
    )

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(
    simPFProducer,
    trackTimeValueMap = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModel"),
    trackTimeErrorMap = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModelResolution"),
    gsfTrackTimeValueMap = cms.InputTag("gsfTrackTimeValueMapProducer:electronGsfTracksConfigurableFlatResolutionModel"),
    gsfTrackTimeErrorMap = cms.InputTag("gsfTrackTimeValueMapProducer:electronGsfTracksConfigurableFlatResolutionModelResolution"),
)
