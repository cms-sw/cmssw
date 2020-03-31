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

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toModify(
    simPFProducer,
    trackTimeValueMap = cms.InputTag("tofPID:t0"),
    trackTimeErrorMap = cms.InputTag("tofPID:sigmat0"),
    #this will cause no time to be set for gsf tracks
    #(since this is not available for the fullsim/reconstruction yet)
    #*TODO* update when gsf times are available
    gsfTrackTimeValueMap = cms.InputTag("tofPID:t0"),
    gsfTrackTimeErrorMap = cms.InputTag("tofPID:sigmat0"),
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(simPFProducer,
    trackingParticleSrc = "mixData:MergedTrackTruth",
    caloParticlesSrc = "mixData:MergedCaloTruth",
    simClusterTruthSrc = "mixData:MergedCaloTruth",
)
