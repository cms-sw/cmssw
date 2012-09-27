import FWCore.ParameterSet.Config as cms

# parameters for HIPAlignmentAlgorithm
HIPAlignmentAlgorithm = cms.PSet(
    applyAPE = cms.bool(True),
    maxRelParameterError = cms.double(1.0),
    outpath = cms.string('./'),
    collectorNJobs = cms.int32(0),
    uvarFile = cms.string('IOUserVariables.root'),
    apeParam = cms.VPSet(cms.PSet(
        function = cms.string('linear'), ## linear or exponential

        apeRPar = cms.vdouble(0.0, 0.0, 3.0),
        apeSPar = cms.vdouble(0.2, 0.0, 3.0),
        Selector = cms.PSet(
            alignParams = cms.vstring('AllAlignables,000000')
        )
    )),
    iterationFile = cms.string('IOIteration.root'),
    collectorActive = cms.bool(False),
    collectorPath = cms.string(''),
    parameterFile = cms.string('IOAlignmentParameters.root'),
    outfile2 = cms.string('HIPAlignmentAlignables.root'),
    algoName = cms.string('HIPAlignmentAlgorithm'),
    trueFile = cms.string('IOTruePositions.root'),
    eventPrescale = cms.int32(20),
    outfile = cms.string('HIPAlignmentEvents.root'),
	surveyFile = cms.string('HIPSurveyResiduals.root'),
    maxAllowedHitPull = cms.double(-1.0),
    surveyResiduals = cms.untracked.vstring(), ## no survey constraint

    misalignedFile = cms.string('IOMisalignedPositions.root'),
    minimumNumberOfHits = cms.int32(50),
    verbosity = cms.bool(False),
    # Dump tracks before and after refit
    debug = cms.bool(False),
    alignedFile = cms.string('IOAlignedPositions.root'),
	fillTrackMonitoring = cms.untracked.bool(False)
)

