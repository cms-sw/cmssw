import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

dqmBeamMonitor = cms.DQMEDAnalyzer("BeamMonitor",
                              monitorName = cms.untracked.string('BeamMonitor'),
                              beamSpot = cms.untracked.InputTag('offlineBeamSpot'), ## hltOfflineBeamSpot for HLTMON
                              primaryVertex = cms.untracked.InputTag('offlinePrimaryVertices'),
                              timeInterval = cms.untracked.int32(920),
                              fitEveryNLumi = cms.untracked.int32(2),
                              resetEveryNLumi = cms.untracked.int32(20),
                              fitPVEveryNLumi = cms.untracked.int32(1),
                              resetPVEveryNLumi = cms.untracked.int32(2),
                              Debug = cms.untracked.bool(False),
                              recordName = cms.untracked.string('BeamSpotOnlineHLTObjectsRcd'),
                              useLockRecords = cms.untracked.bool(False),
                              nLSForUpload = cms.untracked.int32(5),
                              tcdsRecord = cms.untracked.InputTag('tcdsDigis','tcdsRecord'),
                              BeamFitter = cms.PSet(
        			Debug = cms.untracked.bool(False),
        			TrackCollection = cms.untracked.InputTag('generalTracks'), ## ctfWithMaterialTracksP5 for CRAFT
				IsMuonCollection = cms.untracked.bool(False),
                                WriteAscii = cms.untracked.bool(False),
                                AsciiFileName = cms.untracked.string('BeamFit.txt'),
				SaveNtuple = cms.untracked.bool(False),
				SaveFitResults = cms.untracked.bool(False),
				OutputFileName = cms.untracked.string('BeamFit.root'),
                                MinimumPt = cms.untracked.double(1.2),
                                MaximumEta = cms.untracked.double(2.4),
				MaximumImpactParameter = cms.untracked.double(5),
                                MaximumZ = cms.untracked.double(50),
                                MinimumTotalLayers = cms.untracked.int32(11),
                                MinimumPixelLayers = cms.untracked.int32(3),
                                MaximumNormChi2 = cms.untracked.double(2.0),
                                TrackAlgorithm = cms.untracked.vstring('initialStep'), ## ctf,rs,cosmics,initialStep,lowPtTripletStep...; for all algos, leave it blank
                                TrackQuality = cms.untracked.vstring(), ## loose, tight, highPurity...; for all qualities, leave it blank
			        InputBeamWidth = cms.untracked.double(-1.0), ## if -1 use the value calculated by the analyzer
				FractionOfFittedTrks = cms.untracked.double(0.9),
                                MinimumInputTracks = cms.untracked.int32(100),
				deltaSignificanceCut = cms.untracked.double(5)
                                ),
                              dxBin = cms.int32(200),
                              dxMin = cms.double(-1.0),
                              dxMax = cms.double(1.0),
                              
                              vxBin = cms.int32(100),
                              vxMin = cms.double(-.1),
                              vxMax = cms.double(.1),
                              
                              dzBin = cms.int32(80),
                              dzMin = cms.double(-20),
                              dzMax = cms.double(20),
                              
                              phiBin = cms.int32(63),
                              phiMin = cms.double(-3.15),
                              phiMax = cms.double(3.15)
                              )
