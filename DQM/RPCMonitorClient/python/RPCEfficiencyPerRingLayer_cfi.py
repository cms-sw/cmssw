import FWCore.ParameterSet.Config as cms

rpcEfficiencyPerRingLayer = cms.EDAnalyzer("RPCEfficiencyPerRingLayer",
                                           GlobalFolder = cms.untracked.string('RPC/RPCEfficiency/'),
                                           SaveFile = cms.untracked.bool(False),
                                           NameFile = cms.untracked.string('RPCEfficiency.root'),
                                           NumberOfEndcapDisks  = cms.untracked.int32(4),
                                           NumberOfInnermostEndcapRings  = cms.untracked.int32(2)
                                           )
