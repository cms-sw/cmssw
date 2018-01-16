import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
rpcEfficiency = DQMEDAnalyzer('RPCEfficiency',
                               incldt = cms.untracked.bool(True),
                               incldtMB4 = cms.untracked.bool(True),
                               inclcsc = cms.untracked.bool(True),
                               debug = cms.untracked.bool(False),
                               inves = cms.untracked.bool(True),
                               DuplicationCorrection = cms.untracked.int32(1),
                               rangestrips = cms.untracked.double(1.),
                               rangestripsRB4 = cms.untracked.double(4.),
                               MinCosAng = cms.untracked.double(0.99),
                               MaxD = cms.untracked.double(80.0),
                               MaxDrb4 = cms.untracked.double(150.0),
                               cscSegments = cms.InputTag('cscSegments'),
                               dt4DSegments = cms.InputTag('dt4DSegments'),
                               RecHitLabel = cms.InputTag('rpcRecHits'),
                               EffSaveRootFile = cms.untracked.bool(False),
                               EffRootFileName = cms.untracked.string('/tmp/cimmino/RPCEfficiencyFIRST.root'),
                               EffSaveRootFileEventsInterval = cms.untracked.int32(100)
                               )

rpcefficiency = cms.Sequence(rpcEfficiency)


