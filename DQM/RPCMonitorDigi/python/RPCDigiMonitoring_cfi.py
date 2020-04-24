import FWCore.ParameterSet.Config as cms

rpcdigidqm = cms.EDAnalyzer("RPCMonitorDigi",
                            SaveRootFile = cms.untracked.bool(False),
                            RootFileName = cms.untracked.string('RPCMonitorDigi.root'),
                            UseRollInfo =  cms.untracked.bool(False),
                            UseMuon =  cms.untracked.bool(True),
                            MuonPtCut = cms.untracked.double(3.0),
                            MuonEtaCut= cms.untracked.double(1.9),
                            MuonLabel =  cms.InputTag('muons'),
                            ScalersRawToDigiLabel = cms.InputTag('scalersRawToDigi'),
                            RPCFolder = cms.untracked.string('RPC'),
                            GlobalFolder = cms.untracked.string('SummaryHistograms'),
                            RecHitLabel = cms.InputTag("rpcRecHits"),
                         
                            NoiseFolder  = cms.untracked.string("AllHits"),
                            MuonFolder = cms.untracked.string("Muon")
                            )


