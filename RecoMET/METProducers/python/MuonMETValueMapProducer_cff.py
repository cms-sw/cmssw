import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

muonMETValueMapProducer = cms.EDProducer("MuonMETValueMapProducer",
     TrackAssociatorParameterBlock,      
     muonInputTag     = cms.InputTag("muons"),
     beamSpotInputTag = cms.InputTag("offlineBeamSpot"),
     useTrackAssociatorPositions = cms.bool(True),
     useRecHits        = cms.bool(False), #if True, will use deposits in 3x3 recHits
     useHO             = cms.bool(False), #if True, will correct for deposits in HO
     isAlsoTkMu        = cms.bool(True),  #does the mu have to be a tracker mu?
     towerEtThreshold  = cms.double(0.3), #default MET calculated using towers with Et > 0.5 GeV only
     minPt             = cms.double(10.0),#min global Mu Pt is 10 GeV
     maxEta            = cms.double(2.5), #max global |Eta| is 2.4
     maxNormChi2       = cms.double(10.0),#max global chi2/ndof
     maxd0             = cms.double(0.2), #max global d0
     minnHits          = cms.int32(11),   #minimum # of si hits
     minnValidStaHits  = cms.int32(1)     #minimum # of valid hits in the muon system used in the global muon fit
)
muonMETValueMapProducer.TrackAssociatorParameters.useEcal = False
muonMETValueMapProducer.TrackAssociatorParameters.useHcal = False
muonMETValueMapProducer.TrackAssociatorParameters.useHO = False
muonMETValueMapProducer.TrackAssociatorParameters.useCalo = True
muonMETValueMapProducer.TrackAssociatorParameters.useMuon = False
muonMETValueMapProducer.TrackAssociatorParameters.truthMatch = False
