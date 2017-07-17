import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

muonEnergyDepositAnalyzer = cms.EDAnalyzer("MuonEnergyDepositAnalyzer",
                                           MuonServiceProxy,
                                           MuonCollection = cms.InputTag("muons"),
                                           AlgoName = cms.string('muons'),
                                           hadS9SizeMin = cms.double(0.0), 
                                           emSizeMin = cms.double(0.0),
                                           emS9SizeBin = cms.int32(100),
                                           emS9SizeMin = cms.double(0.0),
                                           hoSizeMax = cms.double(4.0),
                                           hoS9SizeBin = cms.int32(100),
                                           hoSizeMin = cms.double(0.0),
                                           emSizeMax = cms.double(4.0),
                                           hadS9SizeMax = cms.double(10.0),
                                           hoS9SizeMin = cms.double(0.0),
                                           hadSizeMin = cms.double(0.0),
                                           emSizeBin = cms.int32(100),
                                           hadS9SizeBin = cms.int32(200),
                                           debug = cms.bool(False),
                                           emS9SizeMax = cms.double(4.0),
                                           hoS9SizeMax = cms.double(4.0),
                                           hadSizeMax = cms.double(10.0),
                                           hoSizeBin = cms.int32(100),
                                           hadSizeBin = cms.int32(200)
                                           )
