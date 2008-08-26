import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

corMetMuons = cms.EDProducer("MuonMET",
     TrackAssociatorParameterBlock,      
     metTypeInputTag = cms.InputTag("CaloMET"),
     uncorMETInputTag = cms.InputTag("met"),
     muonsInputTag  = cms.InputTag("goodMuonsforMETCorrection"),
     useTrackAssociatorPositions = cms.bool(True)
)
