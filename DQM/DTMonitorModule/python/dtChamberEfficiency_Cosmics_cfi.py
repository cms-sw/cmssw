import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtEfficiencyMonitor = DQMEDAnalyzer('DTChamberEfficiency',
    MuonServiceProxy,
    debug = cms.untracked.bool(True),
    TrackCollection = cms.untracked.InputTag('cosmicMuons'),                                 
    theMaxChi2 = cms.double(100.),
    theNSigma = cms.double(3.),
    theMinNrec = cms.double(20.),
    dt4DSegments = cms.untracked.InputTag('dt4DSegments'),
    theRPCRecHits = cms.untracked.InputTag('dummy'),
    cscSegments = cms.untracked.InputTag('dummy'),
    RPCLayers = cms.bool(False),
    NavigationType = cms.string('Direct')
)

