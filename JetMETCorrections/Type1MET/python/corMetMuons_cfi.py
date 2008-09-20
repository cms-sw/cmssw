import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

corMetMuons = cms.EDProducer("MuonMET",
     TrackAssociatorParameterBlock,      
     metTypeInputTag = cms.InputTag("CaloMET"),
     uncorMETInputTag = cms.InputTag("met"),
     muonsInputTag  = cms.InputTag("goodMuonsforMETCorrection"),
     useTrackAssociatorPositions = cms.bool(True),
     useRecHits = cms.bool(False), #if True, will use deposits in 3x3 recHits
     useHO      = cms.bool(False), #if True, will correct for deposits in HO
     towerEtThreshold = cms.double(0.5) #default MET calculated using towers with Et > 0.5 GeV only
)
