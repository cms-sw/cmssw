import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

muonTiming = DQMStep1Module('MuonTiming',
                                  MuonServiceProxy, 
                                  MuonCollection       = cms.InputTag("muons"),
                                  # histograms parameters
                                  tnbins = cms.int32(40),
                                  tnbinsrpc = cms.int32(20),
                                  terrnbins = cms.int32(20),
                                  terrnbinsrpc = cms.int32(20),
                                  ndofnbins = cms.int32(40),
                                  ptnbins = cms.int32(20),
                                  etanbins = cms.int32(20),
                                  tmax = cms.double(300),
                                  tmaxrpc = cms.double(30),
                                  terrmax = cms.double(10),
                                  terrmaxrpc = cms.double(10),
                                  ndofmax = cms.double(40),
                                  ptmax = cms.double(200),
                                  etamax = cms.double(2.4),
                                  tmin = cms.double(-300),
                                  tminrpc = cms.double(-30),
                                  terrmin = cms.double(0),
                                  terrminrpc = cms.double(0),
                                  ndofmin = cms.double(0),
                                  ptmin = cms.double(0),
                                  etamin = cms.double(-2.4),
                                  etaBarrelMin = cms.double(0),
                                  etaBarrelMax = cms.double(1.1),
                                  etaOverlapMin = cms.double(0.9),
                                  etaOverlapMax = cms.double(1.1),
                                  etaEndcapMin = cms.double(0.9),
                                  etaEndcapMax = cms.double(2.4),
                                  folder = cms.string("Muons/MuonTiming")
                           )

muonTiming_miniAOD = DQMStep1Module('MuonTiming',
                                  MuonServiceProxy,
                                  MuonCollection       = cms.InputTag("slimmedMuons"),
                                  # histograms parameters
                                  tnbins = cms.int32(40),
                                  tnbinsrpc = cms.int32(20),
                                  terrnbins = cms.int32(20),
                                  terrnbinsrpc = cms.int32(20),
                                  ndofnbins = cms.int32(40),
                                  ptnbins = cms.int32(20),
                                  etanbins = cms.int32(20),
                                  tmax = cms.double(300),
                                  tmaxrpc = cms.double(30),
                                  terrmax = cms.double(10),
                                  terrmaxrpc = cms.double(10),
                                  ndofmax = cms.double(40),
                                  ptmax = cms.double(200),
                                  etamax = cms.double(2.4),
                                  tmin = cms.double(-300),
                                  tminrpc = cms.double(-30),
                                  terrmin = cms.double(0),
                                  terrminrpc = cms.double(0),
                                  ndofmin = cms.double(0),
                                  ptmin = cms.double(0),
                                  etamin = cms.double(-2.4),
                                  etaBarrelMin = cms.double(0),
                                  etaBarrelMax = cms.double(1.1),
                                  etaOverlapMin = cms.double(0.9),
                                  etaOverlapMax = cms.double(1.1),
                                  etaEndcapMin = cms.double(0.9),
                                  etaEndcapMax = cms.double(2.4),
                                  folder = cms.string("Muons_miniAOD/MuonTiming")
                           )



