import FWCore.ParameterSet.Config as cms

_category = cms.optional.untracked.PSetTemplate(
    reportEvery = cms.untracked.int32(1),
    limit = cms.optional.untracked.int32,
    timespan = cms.optional.untracked.int32
)

_destination_base = cms.untracked.PSet(
    noLineBreaks = cms.optional.untracked.bool,
    noTimeStamps = cms.optional.untracked.bool,
    lineLength = cms.optional.untracked.int32,
    threshold = cms.optional.untracked.string,
    statisticsThreshold = cms.optional.untracked.string,
    allowAnyLabel_ = _category
)
_destination_no_stat = _destination_base.clone(
    enableStatistics = cms.untracked.bool(False),
    resetStatistics = cms.untracked.bool(False)
)

_file_destination = cms.optional.untracked.PSetTemplate(
    noLineBreaks = cms.optional.untracked.bool,
    noTimeStamps = cms.optional.untracked.bool,
    lineLength = cms.optional.untracked.int32,
    threshold = cms.optional.untracked.string,
    statisticsThreshold = cms.optional.untracked.string,
    enableStatistics = cms.untracked.bool(False),
    resetStatistics = cms.untracked.bool(False),
    filename = cms.optional.untracked.string,
    extension = cms.optional.untracked.string,
    output = cms.optional.untracked.string,
    allowAnyLabel_ = _category
)

_default_pset = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1),
    limit = cms.optional.untracked.int32,
    timespan = cms.optional.untracked.int32,

    noLineBreaks = cms.untracked.bool(False),
    noTimeStamps = cms.untracked.bool(False),
    lineLength = cms.untracked.int32(80),
    threshold = cms.untracked.string("INFO"),
    statisticsThreshold = cms.untracked.string("INFO"),
    allowAnyLabel_ = _category
)


MessageLogger = cms.Service("MessageLogger",
    suppressWarning = cms.untracked.vstring(),
    suppressFwkInfo = cms.untracked.vstring(),
    suppressInfo = cms.untracked.vstring(),
    suppressDebug = cms.untracked.vstring(),
    debugModules = cms.untracked.vstring(),
    cout = _destination_no_stat.clone(
        enable = cms.untracked.bool(False)
        ),
    default = _default_pset.clone(),
    cerr = _destination_base.clone(
        enable = cms.untracked.bool(True),
        enableStatistics = cms.untracked.bool(True),
        resetStatistics = cms.untracked.bool(False),
        statisticsThreshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(5)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkSummary = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO')
    ),
    files = cms.untracked.PSet(
        allowAnyLabel_ = _file_destination
    ),
    allowAnyLabel_ = _category
)


