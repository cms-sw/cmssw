import FWCore.ParameterSet.Config as cms

rpcdigidqm = cms.EDAnalyzer("RPCMonitorDigi",
                            moduleLogName = cms.untracked.string('DigiModule'),
                            SaveRootFile = cms.untracked.bool(False),
                            RootFileNameDigi = cms.untracked.string('RPCMonitorDigi.root'),
                            MuonPtCut = cms.untracked.double(3.0),
                            MuonEtaCut= cms.untracked.double(1.6),
                            RPCFolder = cms.untracked.string('RPC'),
                            GlobalFolder = cms.untracked.string('SummaryHistograms'),
                            MuonLabel =  cms.untracked.string('muons'),
                            Noise =  cms.untracked.bool(False),
                            Muon =  cms.untracked.bool(True)
                            )


