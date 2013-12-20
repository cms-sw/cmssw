import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
L2MuonSeeds = cms.EDProducer("L2MuonSeedGenerator",
    MuonServiceProxy,
    L1MinPt = cms.double(0.0),
    InputObjects = cms.InputTag("l1extraParticles"),
    L1MaxEta = cms.double(2.5),
    L1MinQuality = cms.uint32(1),
    GMTReadoutCollection = cms.InputTag("gmtDigis"),
    Propagator = cms.string('SteppingHelixPropagatorAny'),
    UseOfflineSeed = cms.untracked.bool(True),
    UseUnassociatedL1 = cms.bool( True ), 
    OfflineSeedLabel = cms.untracked.InputTag("L2OfflineMuonSeeds")
)



