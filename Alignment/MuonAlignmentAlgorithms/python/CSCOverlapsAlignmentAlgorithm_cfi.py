import FWCore.ParameterSet.Config as cms

import Alignment.MuonAlignmentAlgorithms.CSCOverlapsAlignmentAlgorithm_ringfitters_cff

CSCOverlapsAlignmentAlgorithm = cms.PSet(
    algoName = cms.string("CSCOverlapsAlignmentAlgorithm"),

    # alignment mode: phiy, phipos, phiz
    mode = cms.string("phipos"),

    # global control and output file names
    reportFileName = cms.string("report.py"),
    writeTemporaryFile = cms.string(""),
    readTemporaryFiles = cms.vstring("test.tmp"),
    doAlignment = cms.bool(True),
    makeHistograms = cms.bool(True),

    # selection and fitting parameters
    minP = cms.double(5.),
    minHitsPerChamber = cms.int32(5),
    maxdrdz = cms.double(0.2),
    maxRedChi2 = cms.double(10.),
    fiducial = cms.bool(True),
    useHitWeights = cms.bool(True),
    truncateSlopeResid = cms.double(30.),
    truncateOffsetResid = cms.double(15.),
    combineME11 = cms.bool(True),
    useTrackWeights = cms.bool(False),
    errorFromRMS = cms.bool(False),
    minTracksPerOverlap = cms.int32(10),

    # if we refit tracks using the standard refitter (for dphi/dz track slopes), we need a configured TrackTransformer
    slopeFromTrackRefit = cms.bool(False),
    minStationsInTrackRefits = cms.int32(2),
    TrackTransformer = cms.PSet(DoPredictionsOnly = cms.bool(False),
                                Fitter = cms.string("KFFitterForRefitInsideOut"),
                                TrackerRecHitBuilder = cms.string("WithoutRefit"),
                                Smoother = cms.string("KFSmootherForRefitInsideOut"),
                                MuonRecHitBuilder = cms.string("MuonRecHitBuilder"),
                                RefitDirection = cms.string("alongMomentum"),
                                RefitRPCHits = cms.bool(False),
                                Propagator = cms.string("SteppingHelixPropagatorAny")),
    
    fitters = Alignment.MuonAlignmentAlgorithms.CSCOverlapsAlignmentAlgorithm_ringfitters_cff.fitters,
    )
