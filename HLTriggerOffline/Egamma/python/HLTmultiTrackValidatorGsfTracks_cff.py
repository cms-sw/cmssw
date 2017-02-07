import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.HLTmultiTrackValidator_cfi import *
hltGsfTrackValidator = hltMultiTrackValidator.clone(
    label = [
        "hltEgammaGsfTracks",
        "hltEgammaGsfTracksUnseeded",
    ],
    label_tp_effic           = "trackingParticlesElectron",
    label_tp_effic_refvector = cms.bool(True), 
    dirName        = cms.string('HLT/EG/Tracking/ValidationWRTtp/')
)

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
