import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock
#from JetMETCorrections.TauJet.TCTauAlgo_cfi import *

tcRecoTauProducer = cms.EDProducer("TCRecoTauProducer",
	CaloRecoTauProducer = cms.InputTag("caloRecoTauProducer"),
        ## TauJet jet energy correction parameters
#        src            = cms.InputTag("iterativeCone5CaloJets"),
#        tagName        = cms.string("IterativeCone0.4_EtScheme_TowerEt0.5_E0.8_Jets871_2x1033PU_tau"),
#        TauTriggerType = cms.int32(1),
#	TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters	
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
        TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters
)


