#import FWCore.ParameterSet.Config as cms
from .Types import *
from .Modules import Service

_category = optional.untracked.PSetTemplate(
    reportEvery = untracked.int32(1),
    limit = optional.untracked.int32,
    timespan = optional.untracked.int32
)

_destination_base = untracked.PSet(
    noLineBreaks = optional.untracked.bool,
    noTimeStamps = optional.untracked.bool,
    lineLength = optional.untracked.int32,
    threshold = optional.untracked.string,
    statisticsThreshold = optional.untracked.string,
    allowAnyLabel_ = _category
)
_destination_no_stat = _destination_base.clone(
    enableStatistics = untracked.bool(False),
    resetStatistics = untracked.bool(False)
)

_file_destination = optional.untracked.PSetTemplate(
    noLineBreaks = optional.untracked.bool,
    noTimeStamps = optional.untracked.bool,
    lineLength = optional.untracked.int32,
    threshold = optional.untracked.string,
    statisticsThreshold = optional.untracked.string,
    enableStatistics = untracked.bool(False),
    resetStatistics = untracked.bool(False),
    filename = optional.untracked.string,
    extension = optional.untracked.string,
    output = optional.untracked.string,
    allowAnyLabel_ = _category
)

_default_pset = untracked.PSet(
    reportEvery = untracked.int32(1),
    limit = optional.untracked.int32,
    timespan = optional.untracked.int32,

    noLineBreaks = untracked.bool(False),
    noTimeStamps = untracked.bool(False),
    lineLength = untracked.int32(80),
    threshold = untracked.string("INFO"),
    statisticsThreshold = untracked.string("INFO"),
    allowAnyLabel_ = _category
)


MessageLogger = Service("MessageLogger",
    suppressWarning = untracked.vstring(),
    suppressFwkInfo = untracked.vstring(),
    suppressInfo = untracked.vstring(),
    suppressDebug = untracked.vstring(),
    debugModules = untracked.vstring(),
    cout = _destination_no_stat.clone(
        enable = untracked.bool(False)
        ),
    default = _default_pset.clone(),
    cerr = _destination_base.clone(
        enable = untracked.bool(True),
        enableStatistics = untracked.bool(False),
        resetStatistics = untracked.bool(False),
        statisticsThreshold = untracked.string('WARNING'),
        INFO = untracked.PSet(
            limit = untracked.int32(0)
        ),
        noTimeStamps = untracked.bool(False),
        FwkReport = untracked.PSet(
            reportEvery = untracked.int32(1),
            limit = untracked.int32(10000000)
        ),
        default = untracked.PSet(
            limit = untracked.int32(10000000)
        ),
        Root_NoDictionary = untracked.PSet(
            limit = untracked.int32(0)
        ),
        FwkSummary = untracked.PSet(
            reportEvery = untracked.int32(1),
            limit = untracked.int32(10000000)
        ),
        threshold = untracked.string('INFO')
    ),
    files = untracked.PSet(
        allowAnyLabel_ = _file_destination
    ),
    allowAnyLabel_ = _category
)


