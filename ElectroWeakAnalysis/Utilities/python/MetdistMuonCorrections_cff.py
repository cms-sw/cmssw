import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

distMetGlobalMuons = cms.EDProducer("MuonMET",
     metTypeInputTag = cms.InputTag("CaloMET"),
     uncorMETInputTag = cms.InputTag("met"),
     muonsInputTag  = cms.InputTag("distortedMuons"),
     muonMETDepositValueMapInputTag = cms.InputTag("distmuonMETValueMapProducer","muCorrData","")
)
