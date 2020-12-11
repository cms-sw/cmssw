import FWCore.ParameterSet.Config as cms

ppsStraightTrackAligner = cms.EDAnalyzer("PPSStraightTrackAligner",
    verbosity = cms.untracked.uint32(1),

    # ---------- input and event selection ----------

    tagUVPatternsStrip = cms.InputTag("totemRPUVPatternFinder"),
    tagDiamondHits = cms.InputTag("ctppsFastLocalSimulation"),
    tagPixelHits = cms.InputTag("ctppsFastLocalSimulation"),
    tagPixelLocalTracks = cms.InputTag(""),

    # list of RPs for which the alignment parameters shall be optimized
    rpIds = cms.vuint32(),

    # list of planes to be excluded from processing
    excludePlanes = cms.vuint32(),

    # maximum number of selected events
    maxEvents = cms.int32(-1),  # -1 means unlimited


    # ---------- event selection ----------

    # parameters of hit outlier removal
    maxResidualToSigma = cms.double(3),
    minimumHitsPerProjectionPerRP = cms.uint32(4),

    # skip events with hits in both top and bottom RPs
    removeImpossible = cms.bool(True),

    # minimum required number of units active
    requireNumberOfUnits = cms.uint32(2),

    # require combination of top+horizontal or bottom+horizontal RPs
    requireOverlap = cms.bool(False),

    # require at least 3 RPs active when track in the horizontal-vertical overlap
    requireAtLeast3PotsInOverlap = cms.bool(True),

    # list of RP sets that are accepted irrespective of the "require" settings
    #     the sets should be separated by ";"
    #     within each set, RP ids are separated by ","
    #     example: "103,104;120,121"
    additionalAcceptedRPSets = cms.string(""),

    # track fit quality requirements
    cutOnChiSqPerNdf = cms.bool(True),
    chiSqPerNdfCut = cms.double(10),

    # track angular requirements
    maxTrackAx = cms.double(1E6),
    maxTrackAy = cms.double(1E6),


    # ---------- constraints ----------

    # choices: fixedDetectors, standard
    constraintsType = cms.string("standard"),

    oneRotZPerPot = cms.bool(False),
    useEqualMeanUMeanVRotZConstraints = cms.bool(True),
    
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

    standardConstraints = cms.PSet(
        units = cms.vuint32(1, 21)
    ),


    # ---------- solution parameters ----------

    # a characteristic z in mm
    z0 = cms.double(0.0),

    # what to be resolved
    resolveShR = cms.bool(True),
    resolveShZ = cms.bool(False),
    resolveRotZ = cms.bool(False),

    # suitable value for station alignment
    singularLimit = cms.double(1E-8),


    # ---------- algorithm configuration ----------

    # available algorithms: Ideal, Jan
    algorithms = cms.vstring("Jan"),
    
    IdealResult = cms.PSet(
    ),

    JanAlignmentAlgorithm = cms.PSet(
      weakLimit = cms.double(1E-6),
      stopOnSingularModes = cms.bool(True),
      buildDiagnosticPlots = cms.bool(True),
    ),


    # ---------- output configuration ----------

    saveIntermediateResults = cms.bool(True),
    taskDataFileName = cms.string(''),

    buildDiagnosticPlots = cms.bool(True),
    diagnosticsFile = cms.string(''),

    fileNamePrefix = cms.string('results_iteration_'),
    expandedFileNamePrefix = cms.string('results_cumulative_expanded_'),
    factoredFileNamePrefix = cms.string('results_cumulative_factored_'),
    preciseXMLFormat = cms.bool(False),
    saveXMLUncertainties = cms.bool(False)
)
