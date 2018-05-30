import FWCore.ParameterSet.Config as cms

alcaBeamSpotProducerHP = cms.EDProducer("AlcaBeamSpotProducer",
    AlcaBeamSpotProducerParameters = cms.PSet(
        RunAllFitters = cms.bool(False), ## False: run only default fitter
        RunBeamWidthFit = cms.bool(False), 
        WriteToDB = cms.bool(False), ## do not write results to DB
        fitEveryNLumi = cms.untracked.int32( 1 ),
        resetEveryNLumi = cms.untracked.int32( 1 )
    ),
    BeamFitter = cms.PSet(
        Debug = cms.untracked.bool(False),
        TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias'),
        IsMuonCollection = cms.untracked.bool(False),
        WriteAscii = cms.untracked.bool(False),
        AsciiFileName = cms.untracked.string('BeamFit.txt'), ## all results
        AppendRunToFileName = cms.untracked.bool(True), #runnumber will be inserted to the file name
        WriteDIPAscii = cms.untracked.bool(False),
        DIPFileName = cms.untracked.string('BeamFitDIP.txt'), ## only the last results, for DIP
        SaveNtuple = cms.untracked.bool(False),
        SaveFitResults = cms.untracked.bool(False),
        SavePVVertices = cms.untracked.bool(False),
        OutputFileName = cms.untracked.string('analyze_d0_phi.root'),
        MinimumPt = cms.untracked.double(1.0),
        MaximumEta = cms.untracked.double(2.4),
        MaximumImpactParameter = cms.untracked.double(1.0),
        MaximumZ = cms.untracked.double(60),
        MinimumTotalLayers = cms.untracked.int32(6),
        MinimumPixelLayers = cms.untracked.int32(-1),
        MaximumNormChi2 = cms.untracked.double(10),
        TrackAlgorithm = cms.untracked.vstring(), ## ctf,rs,cosmics,initialStep,lowPtTripletStep...; for all algos, leave it blank
        TrackQuality = cms.untracked.vstring(), ## loose, tight, highPurity...; for all qualities, leave it blank
        InputBeamWidth = cms.untracked.double(-1.0), ## if -1 use the value calculated by the analyzer
        FractionOfFittedTrks = cms.untracked.double(0.9),
        MinimumInputTracks = cms.untracked.int32(50)
     ),
     PVFitter = cms.PSet(
        Debug = cms.untracked.bool(False),
        Apply3DFit = cms.untracked.bool(True),
        VertexCollection = cms.untracked.InputTag('offlinePrimaryVertices'),
        #WriteAscii = cms.untracked.bool(True),
        #AsciiFileName = cms.untracked.string('PVFit.txt'),
        maxNrStoredVertices = cms.untracked.uint32(10000),
        minNrVerticesForFit = cms.untracked.uint32(10),
        minVertexNdf = cms.untracked.double(10.),
        maxVertexNormChi2 = cms.untracked.double(10.),
        minVertexNTracks = cms.untracked.uint32(30),
        minVertexMeanWeight = cms.untracked.double(0.5),
        maxVertexR = cms.untracked.double(2),
        maxVertexZ = cms.untracked.double(10),
        errorScale = cms.untracked.double(1.1),
        nSigmaCut = cms.untracked.double(50.),
        FitPerBunchCrossing = cms.untracked.bool(False),
        useOnlyFirstPV = cms.untracked.bool(True),
        minSumPt = cms.untracked.double(50.)
     )
)

