import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ZCounting = DQMEDAnalyzer('ZCounting',
                          TriggerEvent=cms.InputTag(
                              'hltTriggerSummaryAOD', '', 'HLT'),
                          TriggerResults=cms.InputTag(
                              'TriggerResults', '', 'HLT'),
                          edmPVName=cms.untracked.string(
                              'offlinePrimaryVertices'),
                          edmName=cms.untracked.string('muons'),
                          StandaloneReg=cms.untracked.string('standAloneMuons'), # regular standalone track collection
                          StandaloneUpd=cms.untracked.string('standAloneMuons:UpdatedAtVtx'), # updated standalone track collection 
                          edmTrackName=cms.untracked.string('generalTracks'),

                          MuonTriggerNames=cms.vstring("HLT_IsoMu24_v*"),

                          # Tight, Medium, Loose, CustomTight
                          IDType=cms.untracked.string("CustomTight"),
                          IsoType=cms.untracked.string(
                              "NULL"),  # Tracker-based, PF-based
                          # {0.05, 0.10} for Tracker-based, {0.15, 0.25} for PF-based
                          IsoCut=cms.untracked.double(0.),

                          PtCutL1=cms.untracked.double(27.0),
                          PtCutL2=cms.untracked.double(27.0),
                          EtaCutL1=cms.untracked.double(2.4),
                          EtaCutL2=cms.untracked.double(2.4),

                          MassBin=cms.untracked.int32(80),
                          MassMin=cms.untracked.double(50.0),
                          MassMax=cms.untracked.double(130.0),

                          LumiBin=cms.untracked.int32(2500),
                          LumiMin=cms.untracked.double(0.5),
                          LumiMax=cms.untracked.double(2500.5),

                          PVBin=cms.untracked.int32(100),
                          PVMin=cms.untracked.double(0.5),
                          PVMax=cms.untracked.double(100.5),

                          VtxNTracksFitMin=cms.untracked.double(0.),
                          VtxNdofMin=cms.untracked.double(4.),
                          VtxAbsZMax=cms.untracked.double(24.),
                          VtxRhoMax=cms.untracked.double(2.)
                          )
