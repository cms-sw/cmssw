import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

dqmBeamMonitor_pixelless = DQMEDAnalyzer("BeamMonitor",
                              monitorName = cms.untracked.string('BeamMonitor_PixelLess'),
                              beamSpot = cms.untracked.InputTag('offlineBeamSpot'), ## hltOfflineBeamSpot for HLTMON
                              fitEveryNLumi = cms.untracked.int32(5),
                              resetEveryNLumi = cms.untracked.int32(40),
                              fitPVEveryNLumi = cms.untracked.int32(1),
                              resetPVEveryNLumi = cms.untracked.int32(2),
                              Debug = cms.untracked.bool(False),
                              BeamFitter = cms.PSet(
        			Debug = cms.untracked.bool(False),
        			TrackCollection = cms.untracked.InputTag('ctfPixelLess'),
				IsMuonCollection = cms.untracked.bool(False),
                                WriteAscii = cms.untracked.bool(False),
                                AsciiFileName = cms.untracked.string('BeamFit.txt'),
				SaveNtuple = cms.untracked.bool(False),
				SaveFitResults = cms.untracked.bool(False),
				OutputFileName = cms.untracked.string('BeamFit.root'),
                                MinimumPt = cms.untracked.double(1.),
                                MaximumEta = cms.untracked.double(2.4),
				MaximumImpactParameter = cms.untracked.double(5),
                                MaximumZ = cms.untracked.double(60),
                                MinimumTotalLayers = cms.untracked.int32(6),
                                MinimumPixelLayers = cms.untracked.int32(0),
                                MaximumNormChi2 = cms.untracked.double(5.0),
                                TrackAlgorithm = cms.untracked.vstring(), ## ctf,rs,cosmics,initialStep,lowPtTripletStep...; for all algos, leave it blank
                                TrackQuality = cms.untracked.vstring(), ## loose, tight, highPurity...; for all qualities, leave it blank
			        InputBeamWidth = cms.untracked.double(-1.0), ## if -1 use the value calculated by the analyzer
				FractionOfFittedTrks = cms.untracked.double(0.9),
                                MinimumInputTracks = cms.untracked.int32(80),
				deltaSignificanceCut = cms.untracked.double(10)
                                ),
                              dxBin = cms.int32(400),
                              dxMin = cms.double(-2.0),
                              dxMax = cms.double(2.0),
                              
                              vxBin = cms.int32(500),
                              vxMin = cms.double(-1.0),
                              vxMax = cms.double(1.0),
                              
                              dzBin = cms.int32(120),
                              dzMin = cms.double(-60),
                              dzMax = cms.double(60),
                              
                              phiBin = cms.int32(63),
                              phiMin = cms.double(-3.15),
                              phiMax = cms.double(3.15)
                              )
