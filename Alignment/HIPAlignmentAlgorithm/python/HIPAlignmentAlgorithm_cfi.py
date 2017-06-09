import FWCore.ParameterSet.Config as cms

# parameters for HIPAlignmentAlgorithm
HIPAlignmentAlgorithm = cms.PSet(
    algoName = cms.string('HIPAlignmentAlgorithm'),
    debug = cms.bool(False),
    verbosity = cms.bool(False),
    checkDbAlignmentValidity=cms.bool(False),

    isCollision = cms.bool(True),
    UsePreSelection = cms.bool(False),
    eventPrescale = cms.int32(1),

    multiIOV=cms.bool(False),
    IOVrange=cms.vuint32(1,99999999),

    maxRelParameterError = cms.double(-1), # -1 for no cut
    minimumNumberOfHits = cms.int32(1),
    maxAllowedHitPull = cms.double(-1), # -1 for no cut

    applyCutsPerComponent = cms.bool(False), # Overrides settings above for the specified detectors
    cutsPerComponent = cms.VPSet(
        cms.PSet(
            Selector = cms.PSet(
                alignParams = cms.vstring(
                    "AllAlignables,000000" # Obligatory second string
                ) # can use "selected" for the already-specified alignables
            ),
            maxRelParError = cms.double(-1), # -1 for no cut
            maxHitPull = cms.double(-1), # -1 for no cut
            minNHits = cms.int32(0),
        )
    ),

    # APE settings
    applyAPE = cms.bool(False),
    apeParam = cms.VPSet(
        cms.PSet(
            Selector = cms.PSet(
                alignParams = cms.vstring(
                    "AllAlignables,000000"
                ) # can use "selected" for the already-specified alignables
            ),
            function = cms.string('linear'), ## linear, step or exponential
            apeRPar = cms.vdouble(0, 0, 0), # cm
            apeSPar = cms.vdouble(0, 0, 0), # mrad
        )
    ),

    # Re-weighting
    UseReweighting = cms.bool(False),
    Weight = cms.double(1),
    UniformEta = cms.bool(False),

    # Impact angle cut
    CLAngleCut = cms.double(1.571), # upper bound on collision track impact angle, default -no cut
    CSAngleCut = cms.double(0),  # lower bound on cosmics track impact angle, default -no cut

    # Chisquare scan
    setScanDet = cms.vdouble(0,0,0), # detector ID (1=all det), start,step

    # File paths and names
    collectorActive = cms.bool(False),
    collectorNJobs = cms.int32(0),
    collectorPath = cms.string(''),
    outpath = cms.string('./'),
    uvarFile = cms.string('IOUserVariables.root'),
    alignedFile = cms.string('IOAlignedPositions.root'),
    misalignedFile = cms.string('IOMisalignedPositions.root'),
    trueFile = cms.string('IOTruePositions.root'),
    parameterFile = cms.string('IOAlignmentParameters.root'),
    iterationFile = cms.string('IOIteration.root'),
    outfile2 = cms.string('HIPAlignmentAlignables.root'),

    fillTrackMonitoring = cms.untracked.bool(False),
    outfile = cms.string('HIPAlignmentEvents.root'),

    surveyResiduals = cms.untracked.vstring(), ## no survey constraint
    surveyFile = cms.string('HIPSurveyResiduals.root'),
)

