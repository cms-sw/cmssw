import FWCore.ParameterSet.Config as cms

RPStraightTrackAligner = cms.EDAnalyzer("RPStraightTrackAligner",
    verbosity = cms.untracked.uint32(0),
    factorizationVerbosity = cms.untracked.uint32(0),

    # description of the track-candidate producer module
    #   use `RPSinglTrackCandFind' for parallel finder
    #   use `NonParallelTrackFinder' for non-parallel finder    
    tagRecognizedPatterns = cms.InputTag('NonParallelTrackFinder'),

    # list of RPs for which the alignment parameters shall be optimized
    RPIds = cms.vuint32(),

    # list of planes to be excluded from processing
    excludePlanes = cms.vuint32(),

    # a characteristic z in mm
    z0 = cms.double(0.0),

    resolveShR = cms.bool(True),
    resolveShZ = cms.bool(False),
    resolveRotZ = cms.bool(True),
    resolveRPShZ = cms.bool(False),

    # available algorithms: Ideal, Jan and Millepede
    algorithms = cms.vstring(),

    # suitable value for station alignment
    singularLimit = cms.double(1E-8),

    useExternalFitter = cms.bool(False),
    tagExternalFit = cms.InputTag(''),

    # homogeneous, fixedDetectors, dynamic (still unsupported), final
    #constraintsType = cms.string("homogeneous"),
    constraintsType = cms.string("fixedDetectors"),

    useExtendedRotZConstraint = cms.bool(True),
    useZeroThetaRotZConstraint = cms.bool(False),
    useExtendedShZConstraints = cms.bool(True),
    useExtendedRPShZConstraint = cms.bool(True),
    oneRotZPerPot = cms.bool(False),
    
    # still C^T A values (i.e. not theta values)
    homogeneousConstraints = cms.PSet(
      ShR_values = cms.vdouble(0., 0., 0., 0.),
      ShZ_values = cms.vdouble(0., 0., 0., 0.),
      RotZ_values = cms.vdouble(0., 0.),
      RPShZ_values = cms.vdouble(0., 0.)
    ),
    
    # values in um and m rad
    fixedDetectorsConstraints = cms.PSet(
      ShR = cms.PSet(
        ids = cms.vuint32(1220, 1221, 1228, 1229),
        values = cms.vdouble(0, 0, 0, 0),
      ),
      ShZ = cms.PSet(
        ids = cms.vuint32(1200, 1201, 1208, 1209),
        values = cms.vdouble(0, 0, 0, 0),
      ),
      RotZ = cms.PSet(
        ids = cms.vuint32(1200, 1201),
        values = cms.vdouble(0, 0),
      ),
      RPShZ = cms.PSet(
        ids = cms.vuint32(1200), # number of any plane in the chosen RP
        values = cms.vdouble(0),
      ),
    ),

    maxEvents = cms.uint32(0),  # 0 means unlimited

    maxResidualToSigma = cms.double(3),
    minimumHitsPerProjectionPerRP = cms.uint32(4),

    # skip events with hits in both top and bottom RPs
    removeImpossible = cms.bool(True),

    # minimum required number of units active
    requireNumberOfUnits = cms.uint32(2),

    # require combination of top+horizontal or bottom+horizontal RPs
    requireOverlap = cms.bool(False),

    requireAtLeast3PotsInOverlap = cms.bool(True),

    cutOnChiSqPerNdf = cms.bool(True),
    chiSqPerNdfCut = cms.double(10),

    saveIntermediateResults = cms.bool(True),
    taskDataFileName = cms.string(''),

    diagnosticsFile = cms.string(''),
    buildDiagnosticPlots = cms.bool(True),

    fileNamePrefix = cms.string(''),
    cumulativeFileNamePrefix = cms.string('cumulative_results_'),
    expandedFileNamePrefix = cms.string('cumulative_expanded_results_'),
    factoredFileNamePrefix = cms.string('cumulative_factored_results_'),
    preciseXMLFormat = cms.bool(False),
    
    IdealResult = cms.PSet(
      useExtendedConstraints = cms.bool(True)
    ),

    MillepedeAlgorithm = cms.PSet(
      workingDir = cms.string('/tmp/')
    ),

    JanAlignmentAlgorithm = cms.PSet(
      weakLimit = cms.double(1E-6),
      stopOnSingularModes = cms.bool(True),
      buildDiagnosticPlots = cms.bool(True),
    )
)
