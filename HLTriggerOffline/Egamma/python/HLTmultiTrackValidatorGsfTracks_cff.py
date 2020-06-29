import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.HLTmultiTrackValidator_cfi import *
hltGsfTrackValidator = hltMultiTrackValidator.clone(
    label = [
        "hltEgammaGsfTracks",
#        "hltEgammaGsfTracksUnseeded",
    ],
    label_tp_effic           = "trackingParticlesElectron",
    label_tp_effic_refvector = cms.bool(True), 
    dirName                  = cms.string('HLT/EGM/Tracking/ValidationWRTtp/'),
    ## eta range driven by ECAL acceptance
    histoProducerAlgoBlock = dict(
        TpSelectorForEfficiencyVsEta  = dict(minRapidity=-3, maxRapidity=3),
        TpSelectorForEfficiencyVsPhi  = dict(minRapidity=-3, maxRapidity=3),
        TpSelectorForEfficiencyVsPt   = dict(minRapidity=-3, maxRapidity=3),
        TpSelectorForEfficiencyVsVTXR = dict(minRapidity=-3, maxRapidity=3),
        TpSelectorForEfficiencyVsVTXZ = dict(minRapidity=-3, maxRapidity=3),
        generalTpSelector             = dict(minRapidity=-3, maxRapidity=3),
    ),
    maxRapidityTP =  3.0,
    minRapidityTP = -3.0,
)

from Validation.RecoTrack.TrackValidation_cff import trackingParticlesElectron
hltMultiTrackValidationGsfTracksTask = cms.Task(
   hltTPClusterProducer
   , hltTrackAssociatorByHits
   , trackingParticlesElectron
 )
hltMultiTrackValidationGsfTracks = cms.Sequence(
    hltGsfTrackValidator,
    hltMultiTrackValidationGsfTracksTask
)    
