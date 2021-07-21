import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

dqmBeamMonitor = DQMEDAnalyzer("BeamMonitor",
                              monitorName = cms.untracked.string('BeamMonitor'),
                              beamSpot = cms.untracked.InputTag('offlineBeamSpot'), ## hltOfflineBeamSpot for HLTMON
                              primaryVertex = cms.untracked.InputTag('offlinePrimaryVertices'),
                              timeInterval = cms.untracked.int32(920),
                              fitEveryNLumi = cms.untracked.int32(1),
                              resetEveryNLumi = cms.untracked.int32(20),
                              fitPVEveryNLumi = cms.untracked.int32(1),
                              resetPVEveryNLumi = cms.untracked.int32(5),
                              Debug = cms.untracked.bool(False),
                              OnlineMode = cms.untracked.bool(True),
                              recordName = cms.untracked.string('BeamSpotOnlineHLTObjectsRcd'),
                              useLockRecords = cms.untracked.bool(False),
                              hltResults = cms.InputTag("TriggerResults::HLT"),
                              BeamFitter = cms.PSet(
                                Debug = cms.untracked.bool(False),
                                TrackCollection = cms.untracked.InputTag('generalTracks'),
                                IsMuonCollection = cms.untracked.bool(False),
                                WriteAscii = cms.untracked.bool(False),
                                AsciiFileName = cms.untracked.string('BeamFit.txt'), ## all results
                                AppendRunToFileName = cms.untracked.bool(True), #runnumber will be inserted to the file name
                                WriteDIPAscii = cms.untracked.bool(False),
                                DIPFileName = cms.untracked.string('BeamFitDIP.txt'),
                                SaveNtuple = cms.untracked.bool(False),
                                SavePVVertices = cms.untracked.bool(False),
                                SaveFitResults = cms.untracked.bool(False),
                                OutputFileName = cms.untracked.string('BeamFit.root'), ## ntuple filename
                                MinimumPt = cms.untracked.double(1.0),
                                MaximumEta = cms.untracked.double(2.4),
                                MaximumImpactParameter = cms.untracked.double(1.0),
                                MaximumZ = cms.untracked.double(60),
                                MinimumTotalLayers = cms.untracked.int32(6),
                                MinimumPixelLayers = cms.untracked.int32(0),
                                MaximumNormChi2 = cms.untracked.double(10.0),
                                TrackAlgorithm = cms.untracked.vstring(), ## ctf,rs,cosmics,initialStep,lowPtTripletStep...; for all algos, leave it blank
                                TrackQuality = cms.untracked.vstring(), ## loose, tight, highPurity...; for all qualities, leave it blank
                                InputBeamWidth = cms.untracked.double(0.0060), ## beam width used for Trk fitter, used only when result from PV is not available
                                FractionOfFittedTrks = cms.untracked.double(0.9),
                                MinimumInputTracks = cms.untracked.int32(150),
                                deltaSignificanceCut = cms.untracked.double(10)
                                ),
                              PVFitter = cms.PSet(
                                Debug = cms.untracked.bool(False),
                                Apply3DFit = cms.untracked.bool(True),
                                VertexCollection = cms.untracked.InputTag('offlinePrimaryVertices'),
                                #WriteAscii = cms.untracked.bool(True),
                                #AsciiFileName = cms.untracked.string('PVFit.txt'),
                                maxNrStoredVertices = cms.untracked.uint32(100000),
                                minNrVerticesForFit = cms.untracked.uint32(50),
                                minVertexNdf = cms.untracked.double(10.),
                                maxVertexNormChi2 = cms.untracked.double(10.),
                                minVertexNTracks = cms.untracked.uint32(0),
                                minVertexMeanWeight = cms.untracked.double(0.5),
                                maxVertexR = cms.untracked.double(2),
                                maxVertexZ = cms.untracked.double(10),
                                errorScale = cms.untracked.double(0.9),
                                nSigmaCut = cms.untracked.double(50.),
                                FitPerBunchCrossing = cms.untracked.bool(False),
                                useOnlyFirstPV = cms.untracked.bool(False),
                                minSumPt = cms.untracked.double(0.)
                                ),
                              dxBin = cms.int32(200),
                              dxMin = cms.double(-1.0),
                              dxMax = cms.double(1.0),
                              
                              vxBin = cms.int32(200),
                              vxMin = cms.double(-0.5),
                              vxMax = cms.double(0.5),
                              
                              dzBin = cms.int32(80),
                              dzMin = cms.double(-20),
                              dzMax = cms.double(20),
                              
                              phiBin = cms.int32(63),
                              phiMin = cms.double(-3.15),
                              phiMax = cms.double(3.15)
                              )
