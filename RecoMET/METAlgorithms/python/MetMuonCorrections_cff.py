import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

corMetGlobalMuons = cms.EDProducer("MuonMET",
     metTypeInputTag = cms.InputTag("CaloMET"),
     uncorMETInputTag = cms.InputTag("met"),
     muonsInputTag  = cms.InputTag("muons"),
     muonMETDepositValueMapInputTag = cms.InputTag("muonMETValueMapProducer","muCorrData","")
)
