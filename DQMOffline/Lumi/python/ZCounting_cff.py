import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
zcounting = DQMEDAnalyzer('ZCounting',
                                 TriggerEvent    = cms.InputTag('hltTriggerSummaryAOD','','HLT'),
                                 TriggerResults  = cms.InputTag('TriggerResults','','HLT'),
				 edmPVName       = cms.untracked.string('offlinePrimaryVertices'),
                                 edmName       = cms.untracked.string('muons'),
                                 edmTrackName = cms.untracked.string('generalTracks'),

                                 edmGsfEleName = cms.untracked.string('gedGsfElectrons'),
                                 edmSCName = cms.untracked.string('particleFlowEGamma'),

                                 effAreasConfigFile = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Summer16/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_80X.txt"),

                                 rhoname = cms.InputTag('fixedGridRhoFastjetAll'),
                                 beamspotName = cms.InputTag('offlineBeamSpot'),
                                 conversionsName = cms.InputTag('conversions'),

                                 IDType   = cms.untracked.string("Tight"),# Tight, Medium, Loose
                                 IsoType  = cms.untracked.string("NULL"),  # Tracker-based, PF-based
                                 IsoCut   = cms.untracked.double(0.),     # {0.05, 0.10} for Tracker-based, {0.15, 0.25} for PF-based

                                 PtCutL1  = cms.untracked.double(30.0),
                                 PtCutL2  = cms.untracked.double(30.0),
                                 EtaCutL1 = cms.untracked.double(2.4),
                                 EtaCutL2 = cms.untracked.double(2.4),

                                 PtCutEleTag = cms.untracked.double(40.0),
                                 PtCutEleProbe = cms.untracked.double(35.0),
                                 EtaCutEleTag = cms.untracked.double(2.5),
                                 EtaCutEleProbe = cms.untracked.double(2.5),
                                 MassCutEleLow = cms.untracked.double(80.0),
                                 MassCutEleHigh = cms.untracked.double(100.0),

                                 ElectronIDType = cms.untracked.string("TIGHT"),

                                 MassBin  = cms.untracked.int32(50),
                                 MassMin  = cms.untracked.double(66.0),
                                 MassMax  = cms.untracked.double(116.0),

                                 LumiBin  = cms.untracked.int32(2500),
                                 LumiMin  = cms.untracked.double(0.0),
                                 LumiMax  = cms.untracked.double(2500.0),

                                 PVBin    = cms.untracked.int32(60),
                                 PVMin    = cms.untracked.double(0.0),
                                 PVMax    = cms.untracked.double(60.0),

                                 VtxNTracksFitMin = cms.untracked.double(0.),
                                 VtxNdofMin       = cms.untracked.double(4.),
                                 VtxAbsZMax       = cms.untracked.double(24.),
                                 VtxRhoMax        = cms.untracked.double(2.)
                                 )
