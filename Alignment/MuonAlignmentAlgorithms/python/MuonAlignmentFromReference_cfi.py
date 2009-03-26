import FWCore.ParameterSet.Config as cms

MuonAlignmentFromReference = cms.PSet(
    algoName = cms.string("MuonAlignmentFromReference"),

    # which chambers to include in the track fit (by default, none)
    intrackfit = cms.vstring(),

    # which tracks/hits to accept
    minTrackPt = cms.double(20.),
    maxTrackPt = cms.double(100.),
    minTrackerHits = cms.int32(10),
    maxTrackerRedChi2 = cms.double(10.),
    allowTIDTEC = cms.bool(True),
    minDT13Hits = cms.int32(8),
    minDT2Hits = cms.int32(4),
    minCSCHits = cms.int32(6),
    maxDT13AngleError = cms.double(0.005),
    maxDT2AngleError = cms.double(0.030),
    maxCSCAngleError = cms.double(0.005),

    # for parallel processing on the CAF
    writeTemporaryFile = cms.string(""),
    readTemporaryFiles = cms.vstring(),
    doAlignment = cms.bool(True),                   # turn off fitting in residuals-collection jobs

    # fitting options
    twoBin = cms.bool(True),                        # must be the same as residuals-collection job!
    combineME11 = cms.bool(True),                   # must be the same as residuals-collection job!

    residualsModel = cms.string("powerLawTails"),   # this and the following need not be the same
    DT13fitScattering = cms.bool(True),
    DT13fitZpos = cms.bool(True),
    DT13fitPhiz = cms.bool(True),
    DT2fitScattering = cms.bool(True),
    DT2fitPhiz = cms.bool(True),
    CSCfitScattering = cms.bool(True),
    CSCfitZpos = cms.bool(False),  # not enough sensitivity
    CSCfitPhiz = cms.bool(True),

    # where reporting will go
    reportFileName = cms.string("MuonAlignmentFromReference_report.py"),  # Python-formatted output
    rootDirectory = cms.string("MuonAlignmentFromReference"),
    )
