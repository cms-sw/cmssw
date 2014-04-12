import FWCore.ParameterSet.Config as cms

# File: TCMET.cff
# Author: R. Remington & F. Golf 
# Date: 11.14.2008
#
# Form Track Corrected MET

disttcMet = cms.EDProducer("TCMETProducer",
    alias = cms.string('TCMET'),
    electronInputTag  = cms.InputTag("gsfElectrons"),
    muonInputTag      = cms.InputTag("distortedMuons"),
    trackInputTag     = cms.InputTag("generalTracks"),
    metInputTag       = cms.InputTag("met"),
    beamSpotInputTag  = cms.InputTag("offlineBeamSpot"),
    vertexInputTag    = cms.InputTag("offlinePrimaryVertices"),
    muonDepValueMap   = cms.InputTag("distmuonMETValueMapProducer"  , "muCorrData"),     
    tcmetDepValueMap  = cms.InputTag("distmuonTCMETValueMapProducer", "muCorrData"), 
    pt_min  = cms.double(1.0),
    pt_max  = cms.double(100.),
    eta_max = cms.double(2.65), 
    chi2_max = cms.double(5),
    nhits_min = cms.double(6),
    d0_max = cms.double(0.1),
    ptErr_max = cms.double(0.2),
    track_quality = cms.vint32(2),
    track_algos = cms.vint32(), 
    isCosmics = cms.bool(False),
    rf_type = cms.int32(1),
    correctShowerTracks = cms.bool(False),
    usePvtxd0 = cms.bool(False),
    nMinOuterHits = cms.int32(2),
    usedeltaRRejection = cms.bool(False),
    deltaRShower = cms.double(0.01),
    checkTrackPropagation = cms.bool(False),
    radius = cms.double(130.), 
    zdist  = cms.double(314.),
    corner = cms.double(1.479),
    d0cuta = cms.double(0.015),
    d0cutb = cms.double(0.5),
    maxd0cut = cms.double(0.3),
    chi2_tight_max = cms.double(3.0),
    nhits_tight_min = cms.double(11),
    ptErr_tight_max = cms.double(0.1),
    maxTrackAlgo = cms.int32(8)
)



