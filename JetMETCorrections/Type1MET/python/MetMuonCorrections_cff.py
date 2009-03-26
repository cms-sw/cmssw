import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

corMetGlobalMuons = cms.EDProducer("MuonMET",
     metTypeInputTag = cms.InputTag("CaloMET"),
     uncorMETInputTag = cms.InputTag("met"),
     muonsInputTag  = cms.InputTag("muons"),
     muonValueMapFlagInputTag = cms.InputTag("muonMETValueMapProducer","muCorrFlag",""),
     muonValueMapDeltaXInputTag = cms.InputTag("muonMETValueMapProducer","muCorrDepX",""),
     muonValueMapDeltaYInputTag = cms.InputTag("muonMETValueMapProducer","muCorrDepY","")
)
