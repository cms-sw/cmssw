import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

muonTiming = cms.EDAnalyzer("MuonTiming",
                                  MuonServiceProxy, 
                                  MuonCollection       = cms.InputTag("muons"),
                                  # histograms parameters
                                  tnbins = cms.int32(40),
                                  tnbinsrpc = cms.int32(20),
                                  ndofnbins = cms.int32(20),
                                  ptnbins = cms.int32(20),
                                  etanbins = cms.int32(20),
                                  tmax = cms.double(300),
                                  tmaxrpc = cms.double(50),
                                  ndofmax = cms.double(20),
                                  ptmax = cms.double(200),
                                  etamax = cms.double(2.4),
                                  tmin = cms.double(-300),
                                  tminrpc = cms.double(-50),
                                  ndofmin = cms.double(0),
                                  ptmin = cms.double(0),
                                  etamin = cms.double(-2.4),
                                  etaBarrelMin = cms.double(0),
                                  etaBarrelMax = cms.double(1.2),
                                  etaOverlapMin = cms.double(1.2),
                                  etaOverlapMax = cms.double(1.6),
                                  etaEndcapMin = cms.double(1.6),
                                  etaEndcapMax = cms.double(2.4),
				  folder = cms.string("Muons/MuonTiming")
                           )

muonTiming_miniAOD = cms.EDAnalyzer("MuonTiming",
                                  MuonServiceProxy,
                                  MuonCollection       = cms.InputTag("slimmedMuons"),
                                  # histograms parameters
                                  tnbins = cms.int32(40),
                                  tnbinsrpc = cms.int32(20),
                                  ndofnbins = cms.int32(20),
                                  ptnbins = cms.int32(20),
                                  etanbins = cms.int32(20),
                                  tmax = cms.double(300),
                                  tmaxrpc = cms.double(50),
                                  ndofmax = cms.double(20),
                                  ptmax = cms.double(200),
                                  etamax = cms.double(2.4),
                                  tmin = cms.double(-300),
                                  tminrpc = cms.double(-50),
                                  ndofmin = cms.double(0),
                                  ptmin = cms.double(0),
                                  etamin = cms.double(-2.4),
                                  etaBarrelMin = cms.double(0),
                                  etaBarrelMax = cms.double(1.2),
                                  etaOverlapMin = cms.double(1.2),
                                  etaOverlapMax = cms.double(1.6),
                                  etaEndcapMin = cms.double(1.6),
                                  etaEndcapMax = cms.double(2.4),
				  folder = cms.string("Muons_miniAOD/MuonTiming")
                           )



