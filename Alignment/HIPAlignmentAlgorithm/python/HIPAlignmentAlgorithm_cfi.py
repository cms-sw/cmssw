import FWCore.ParameterSet.Config as cms

# parameters for HIPAlignmentAlgorithm
HIPAlignmentAlgorithm = cms.PSet(
    applyAPE = cms.bool(False),
    maxRelParameterError = cms.double(1000.0),
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
    eventPrescale = cms.int32(1),

#CY:Re-weighting
    isCollision = cms.bool(True),
    UsePreSelection = cms.bool(False),
    UseReweighting = cms.bool(True), 
    Weight = cms.double(1.0),
    UniformEta = cms.bool(False),
#CY:Impact angle cut
    CLAngleCut = cms.double(1.571), #upper bound on collision track impact angle, default -no cut
    CSAngleCut = cms.double(0.0),  #lower bound on cosmics track impact angle, default -no cut
#CY:Scan
    setScanDet = cms.vdouble(0,0,0), #detector ID (1=all det), start,step

    outfile = cms.string('HIPAlignmentEvents.root'),
    surveyFile = cms.string('HIPSurveyResiduals.root'),
    maxAllowedHitPull = cms.double(-1.0),
    surveyResiduals = cms.untracked.vstring(), ## no survey constraint

    misalignedFile = cms.string('IOMisalignedPositions.root'),
    minimumNumberOfHits = cms.int32(50),
    verbosity = cms.bool(False),
		checkDbAlignmentValidity=cms.bool(False),
    # Dump tracks before and after refit
    debug = cms.bool(False),
    alignedFile = cms.string('IOAlignedPositions.root'),
    multiIOV= cms.bool(False),
		IOVrange=cms.vuint32(1,99999999),
    fillTrackMonitoring = cms.untracked.bool(False)
)

