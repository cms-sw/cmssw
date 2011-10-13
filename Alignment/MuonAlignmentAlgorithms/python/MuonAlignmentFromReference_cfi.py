import FWCore.ParameterSet.Config as cms

MuonAlignmentFromReference = cms.PSet(
    algoName = cms.string("MuonAlignmentFromReference"),

    # which chambers to include in the track fit (by default, none)
    reference = cms.vstring(),

    # which tracks/hits to accept
    minTrackPt = cms.double(20.),
    maxTrackPt = cms.double(100.),
    minTrackerHits = cms.int32(10),
    maxTrackerRedChi2 = cms.double(10.),
    allowTIDTEC = cms.bool(True),
    minDT13Hits = cms.int32(8),
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

    residualsModel = cms.string("ROOTVoigt"),       # this and the following need not be the same; you can make these decisions at the alignment stage
    minAlignmentHits = cms.int32(30),
    weightAlignment = cms.bool(True),

    # where reporting will go
    reportFileName = cms.string("MuonAlignmentFromReference_report.py"),  # Python-formatted output

    maxResSlopeY = cms.double(10.)
    )
