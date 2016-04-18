import FWCore.ParameterSet.Config as cms

AlcaBeamMonitor = cms.EDAnalyzer("AlcaBeamMonitor",
                                 MonitorName        = cms.untracked.string('AlcaBeamMonitor'),
                                 PrimaryVertexLabel = cms.untracked.InputTag('offlinePrimaryVertices'),
                                 BeamSpotLabel      = cms.untracked.InputTag('offlineBeamSpot'),
                                 #TrackLabel         = cms.untracked.InputTag('ALCARECOTkAlMinBias'),
                                 TrackLabel         = cms.untracked.InputTag('generalTracks'),
                                 ScalerLabel        = cms.untracked.InputTag('scalerBeamSpot'),
                                 BeamFitter = cms.PSet(
                                   Debug = cms.untracked.bool(False),
                                   #TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias'),
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
                                   maxNrStoredVertices = cms.untracked.uint32(10000),
                                   minNrVerticesForFit = cms.untracked.uint32(50),
                                   minVertexNdf = cms.untracked.double(10.),
                                   maxVertexNormChi2 = cms.untracked.double(10.),
                                   minVertexNTracks = cms.untracked.uint32(0),
                                   minVertexMeanWeight = cms.untracked.double(0.5),
                                   maxVertexR = cms.untracked.double(2),
                                   maxVertexZ = cms.untracked.double(10),
                                   errorScale = cms.untracked.double(0.9),
                                   nSigmaCut = cms.untracked.double(5.),
                                   FitPerBunchCrossing = cms.untracked.bool(False)
                                   ),
                               )

# This customization is needed in the trackingLowPU era to be able to
# compute the beamspot also in the cases in which the pixel detector
# is not included in data-taking, like it was the case for "Quiet
# Beam" collisions on 2016 with run 269207.

from Configuration.StandardSequences.Eras import eras
eras.trackingLowPU.toModify(AlcaBeamMonitor,
                            BeamFitter = dict(MaximumImpactParameter = 5.0,
                                              MinimumInputTracks = 50))
