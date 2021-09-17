import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtEfficiencyMonitor = DQMEDAnalyzer('DTChamberEfficiency',
    MuonServiceProxy,
    debug = cms.untracked.bool(True),
    TrackCollection = cms.InputTag('cosmicMuons'),                                 
    theMaxChi2 = cms.double(100.),
    theNSigma = cms.double(3.),
    theMinNrec = cms.double(20.),
    dt4DSegments = cms.InputTag('dt4DSegments'),
    theRPCRecHits = cms.InputTag('dummy'),
    cscSegments = cms.InputTag('dummy'),
    RPCLayers = cms.bool(False),
    NavigationType = cms.string('Direct')
)

