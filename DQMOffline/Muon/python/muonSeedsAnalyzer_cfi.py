import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
muonSeedsAnalyzer = DQMEDAnalyzer('MuonSeedsAnalyzer',
                                   MuonServiceProxy,
                                   SeedCollection = cms.InputTag("ancientMuonSeed"),
                                   
                                   seedPxyzMin = cms.double(-50.0),
                                   pxyzErrMin = cms.double(-100.0),
                                   phiErrMax = cms.double(3.2),
                                   pxyzErrMax = cms.double(100.0),
                                   RecHitBin = cms.int32(25),
                                   etaErrMin = cms.double(0.0),
                                   seedPtMin = cms.double(0.0),
                                   seedPxyzBin = cms.int32(100),
                                   ThetaBin = cms.int32(100),
                                   RecHitMin = cms.double(0.0),
                                   EtaMin = cms.double(-3.0),
                                   pErrBin = cms.int32(200),
                                   phiErrBin = cms.int32(160),
                                   EtaMax = cms.double(3.0),
                                   etaErrBin = cms.int32(200),
                                   seedPxyzMax = cms.double(50.0),
                                   ThetaMin = cms.double(0.0),
                                   PhiMin = cms.double(-3.2),
                                   pxyzErrBin = cms.int32(100),
                                   RecHitMax = cms.double(25.0),
                                   ThetaMax = cms.double(3.2),
                                   pErrMin = cms.double(0.0),
                                   EtaBin = cms.int32(100),
                                   pErrMax = cms.double(200.0),
                                   seedPtMax = cms.double(200.0),
                                   seedPtBin = cms.int32(1000),
                                   phiErrMin = cms.double(0.0),
                                   PhiBin = cms.int32(100),
                                   debug = cms.bool(False),
                                   etaErrMax = cms.double(0.5),
                                   PhiMax = cms.double(3.2)
                                   )
