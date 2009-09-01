import FWCore.ParameterSet.Config as cms

dqmBeamMonitor = cms.EDFilter("BeamMonitor",
                              monitorName = cms.untracked.string('BeamMonitor'),
                              beamSpot = cms.untracked.string('offlineBeamSpot'), ## hltOfflineBeamSpot for HLTMON
                              fitEveryNLumi = cms.untracked.int32(2),
                              resetEveryNLumi = cms.untracked.int32(20),
                              Debug = cms.untracked.bool(False),
                              BeamFitter = cms.PSet(
        			Debug = cms.untracked.bool(False),
        			TrackCollection = cms.untracked.InputTag('ctfWithMaterialTracksP5'), ## ctfWithMaterialTracksP5 for CRAFT
                                WriteAscii = cms.untracked.bool(False),
                                AsciiFileName = cms.untracked.string('BeamFit.txt'),
                                MinimumPt = cms.untracked.double(1.2),
                                MaximumEta = cms.untracked.double(2.4),
				MaximumImpactParameter = cms.untracked.double(5),
                                MaximumZ = cms.untracked.double(300),
                                MinimumTotalLayers = cms.untracked.int32(0),
                                MinimumPixelLayers = cms.untracked.int32(0),
                                MaximumNormChi2 = cms.untracked.double(100.0),
                                TrackAlgorithm = cms.untracked.vstring(), ## ctf,rs,cosmics,iter0,iter1...; for all algos, leave it blank
                                TrackQuality = cms.untracked.vstring(), ## loose, tight, highPurity...; for all qualities, leave it blank
			        InputBeamWidth = cms.untracked.double(-1.0), ## if -1 use the value calculated by the analyzer
                                FractionOfFittedTrks = cms.untracked.double(0.9),
                                MinimumInputTracks = cms.untracked.int32(10)
                                ),
                              dxBin = cms.int32(200),
                              dxMin = cms.double(-1.0),
                              dxMax = cms.double(1.0),

                              vxBin = cms.int32(100),
                              vxMin = cms.double(-.1),
                              vxMax = cms.double(.1),
                              
                              phiBin = cms.int32(63),
                              phiMin = cms.double(-3.15),
                              phiMax = cms.double(3.15)
                              )
