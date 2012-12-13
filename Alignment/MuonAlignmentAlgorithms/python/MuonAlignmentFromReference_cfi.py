import FWCore.ParameterSet.Config as cms

MuonAlignmentFromReference = cms.PSet(
    algoName = cms.string("MuonAlignmentFromReference"),

    muonCollectionTag = cms.InputTag(""),

    # which chambers to include in the track fit (by default, none)
    reference = cms.vstring(),

    # which tracks/hits to accept
    minTrackPt = cms.double(0.),
    maxTrackPt = cms.double(1000.),
    minTrackP = cms.double(0.),
    maxTrackP = cms.double(1000.),
    maxDxy = cms.double(1000.),
    minTrackerHits = cms.int32(10),
    maxTrackerRedChi2 = cms.double(10.),
    allowTIDTEC = cms.bool(True),
    minNCrossedChambers = cms.int32(3),
    minDT13Hits = cms.int32(7),
    minDT2Hits = cms.int32(4),
    minCSCHits = cms.int32(6),

    # for parallel processing on the CAF
    writeTemporaryFile = cms.string(""),
    readTemporaryFiles = cms.vstring(),
    doAlignment = cms.bool(True),                   # turn off fitting in residuals-collection jobs
    strategy = cms.int32(1),

    # fitting options
    twoBin = cms.bool(True),                        # must be the same as residuals-collection job!
    combineME11 = cms.bool(True),                   # must be the same as residuals-collection job!

    residualsModel = cms.string("pureGaussian2D"),
    minAlignmentHits = cms.int32(30),
    weightAlignment = cms.bool(True),
    useResiduals = cms.string("1100"),
    
    specialFitPatternDT6DOF = cms.string(""),
    specialFitPatternDT5DOF = cms.string(""),
    specialFitPatternCSC = cms.string(""),

    # where reporting will go
    reportFileName = cms.string("MuonAlignmentFromReference_report.py"),  # Python-formatted output

    maxResSlopeY = cms.double(10.),
    
    checkTrackFiduciality = cms.bool(False),
    
    createNtuple = cms.bool(False),
    layersDebugDump = cms.bool(False),
    
    peakNSigma = cms.double(-1.),
    bFieldCorrection = cms.int32(1),
    
    doDT = cms.bool(True),
    doCSC = cms.bool(True)
)
