import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.HLTmultiTrackValidator_cfi import *
hltGsfTrackValidator = hltMultiTrackValidator.clone(
    label = [
        "hltEgammaGsfTracks",
        "hltEgammaGsfTracksUnseeded",
    ],
    label_tp_effic           = "trackingParticlesElectron",
    label_tp_effic_refvector = cms.bool(True), 
    dirName                  = cms.string('HLT/EG/Tracking/ValidationWRTtp/'),
)
hltGsfTrackValidator.histoProducerAlgoBlock.GpSelectorForEfficiencyVsEta.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.GpSelectorForEfficiencyVsEta.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.GpSelectorForEfficiencyVsPhi.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.GpSelectorForEfficiencyVsPhi.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.GpSelectorForEfficiencyVsPt.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.GpSelectorForEfficiencyVsPt.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.GpSelectorForEfficiencyVsVTXR.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.GpSelectorForEfficiencyVsVTXR.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.GpSelectorForEfficiencyVsVTXZ.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.GpSelectorForEfficiencyVsVTXZ.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsEta.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsEta.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPhi.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPhi.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPt.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPt.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXR.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXR.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXZ.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXZ.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.generalGpSelector.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.generalGpSelector.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.generalTpSelector.maxRapidity = cms.double( 3.0)
hltGsfTrackValidator.histoProducerAlgoBlock.generalTpSelector.minRapidity = cms.double(-3.0)
hltGsfTrackValidator.maxRapidityTP = cms.double( 3.0)
hltGsfTrackValidator.minRapidityTP = cms.double(-3.0)
hltGsfTrackValidator.maxEta = cms.double( 3.0)
hltGsfTrackValidator.minEta = cms.double(-3.0)

from Validation.RecoTrack.TrackValidation_cff import trackingParticlesElectron
hltGsfTracksPreValidation = cms.Sequence(
    cms.ignore(trackingParticlesElectron)    
#    trackingParticlesElectron
)

hltMultiTrackValidationGsfTracks = cms.Sequence(
    hltTPClusterProducer
    + hltTrackAssociatorByHits
#    + cms.ignore(trackingParticlesElectron)    
    + hltGsfTracksPreValidation
    + hltGsfTrackValidator
)    
