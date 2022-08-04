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

                                 MuonTriggerNames = cms.vstring("HLT_IsoMu24_v*"),
                                 MuonTriggerObjectNames = cms.vstring("hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p08"),

                                 IDType   = cms.untracked.string("CustomTight"), # Tight, Medium, Loose, CustomTight
                                 IsoType  = cms.untracked.string("NULL"),  # Tracker-based, PF-based
                                 IsoCut   = cms.untracked.double(0.),     # {0.05, 0.10} for Tracker-based, {0.15, 0.25} for PF-based

                                 PtCutL1  = cms.untracked.double(27.0),
                                 PtCutL2  = cms.untracked.double(27.0),
                                 EtaCutL1 = cms.untracked.double(2.4),
                                 EtaCutL2 = cms.untracked.double(2.4),

                                 PtCutEleTag = cms.untracked.double(40.0),
                                 PtCutEleProbe = cms.untracked.double(35.0),
                                 EtaCutEleTag = cms.untracked.double(2.5),
                                 EtaCutEleProbe = cms.untracked.double(2.5),
                                 MassCutEleLow = cms.untracked.double(80.0),
                                 MassCutEleHigh = cms.untracked.double(100.0),

                                 ElectronIDType = cms.untracked.string("TIGHT"),

                                 MassBin  = cms.untracked.int32(80),
                                 MassMin  = cms.untracked.double(50.0),
                                 MassMax  = cms.untracked.double(130.0),

                                 LumiBin  = cms.untracked.int32(2500),
                                 LumiMin  = cms.untracked.double(0.5),
                                 LumiMax  = cms.untracked.double(2500.5),

                                 PVBin    = cms.untracked.int32(100),
                                 PVMin    = cms.untracked.double(0.5),
                                 PVMax    = cms.untracked.double(100.5),

                                 VtxNTracksFitMin = cms.untracked.double(0.),
                                 VtxNdofMin       = cms.untracked.double(4.),
                                 VtxAbsZMax       = cms.untracked.double(24.),
                                 VtxRhoMax        = cms.untracked.double(2.)
                                 )


from Configuration.Eras.Modifier_run2_HLTconditions_2016_cff import run2_HLTconditions_2016
run2_HLTconditions_2016.toModify( zcounting, MuonTriggerNames = cms.vstring("HLT_IsoMu24_v*","HLT_IsoTkMu24_v*"),
                                    MuonTriggerObjectNames = cms.vstring("hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07",
                                                                         "hltL3fL1sMu22L1f0Tkf24QL3trkIsoFiltered0p09"),
                                    PtCutL1  = cms.untracked.double(27.0),
                                    PtCutL2  = cms.untracked.double(27.0)
                       )

from Configuration.Eras.Modifier_run2_HLTconditions_2017_cff import run2_HLTconditions_2017
run2_HLTconditions_2017.toModify(zcounting, MuonTriggerNames = cms.vstring("HLT_IsoMu27_v*"),
                                   MuonTriggerObjectNames = cms.vstring("hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07"),
                                   PtCutL1  = cms.untracked.double(30.0),
                                   PtCutL2  = cms.untracked.double(30.0)
                       )

from Configuration.Eras.Modifier_run2_HLTconditions_2018_cff import run2_HLTconditions_2018
run2_HLTconditions_2018.toModify(zcounting, MuonTriggerNames = cms.vstring("HLT_IsoMu24_v*"),
                                   MuonTriggerObjectNames = cms.vstring("hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07"),
                                   PtCutL1  = cms.untracked.double(27.0),
                                   PtCutL2  = cms.untracked.double(27.0)
                       )
