import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.HLTmultiTrackValidator_cfi import *
hltMuonTrackValidator = hltMultiTrackValidator.clone(
    label = [
        "hltIter0HighPtTkMuMerged2016Tk",
    ],
    label_tp_effic           = "trackingParticlesMuon",
    label_tp_effic_refvector = True,
    dirName                  = 'HLT/EG/Tracking/ValidationWRTtp/',
    ## eta range driven by ECAL acceptance
    histoProducerAlgoBlock = dict(
        TpSelectorForEfficiencyVsEta  = dict(ptMin = 24),
        TpSelectorForEfficiencyVsPhi  = dict(ptMin = 24),
        TpSelectorForEfficiencyVsVTXR = dict(ptMin = 24),
        TpSelectorForEfficiencyVsVTXZ = dict(ptMin = 24),
        generalTpSelector             = dict(ptMin = 24),
    ),
)

from Validation.RecoTrack.TrackValidation_cff import trackingParticlesElectron
trackingParticlesMuon = trackingParticlesElectron.clone(pdgId = [-13, 13])
hltMultiTrackValidationMuonTracks = cms.Sequence(
    hltTPClusterProducer
    + hltTrackAssociatorByHits
    + cms.ignore(trackingParticlesMuon)
    + hltMuonTrackValidator
)
