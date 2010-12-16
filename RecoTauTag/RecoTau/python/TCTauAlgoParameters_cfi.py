import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

tcTauAlgoParameters = cms.PSet(
        CaloRecoTauProducer = cms.InputTag("JPTCaloRecoTauProducer"),
        EtCaloOverTrackMin    = cms.double(-0.9),
        EtCaloOverTrackMax    = cms.double(0.0),
        EtHcalOverTrackMin    = cms.double(-0.3),
        EtHcalOverTrackMax    = cms.double(1.0),
        SignalConeSize        = cms.double(0.2),
        EcalConeSize          = cms.double(0.5),
        MatchingConeSize      = cms.double(0.1),
        Track_minPt           = cms.double(1.0),
        tkmaxipt              = cms.double(0.03),
        tkmaxChi2             = cms.double(100.),
        tkminPixelHitsn       = cms.int32(2),
        tkminTrackerHitsn     = cms.int32(8),
        TrackCollection       = cms.InputTag("generalTracks"),
        PVProducer            = cms.InputTag("offlinePrimaryVertices"),
        EBRecHitCollection    = cms.InputTag("ecalRecHit:EcalRecHitsEB"),
        EERecHitCollection    = cms.InputTag("ecalRecHit:EcalRecHitsEE"),
        HBHERecHitCollection  = cms.InputTag("hbhereco"),
        HORecHitCollection    = cms.InputTag("horeco"),
        HFRecHitCollection    = cms.InputTag("hfreco"),
        TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters,
        DropCaloJets          = cms.untracked.bool(False), 
        DropRejectedJets      = cms.untracked.bool(False)
)
