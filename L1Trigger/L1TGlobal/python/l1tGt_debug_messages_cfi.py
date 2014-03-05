#
# Send DEBUG messages from L1T to file l1t_debug.log
#
# NOTE:  to receive debug messages you must have compiled like this:
#
#   scram b -j 8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
#
import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service(
    "MessageLogger",
    destinations       =  cms.untracked.vstring(
                           'cout','l1t_debug',
                           'warnings', 
                           'errors', 
                           'infos', 
                           'debugs',
                           'cerr'
                           ),
    categories         = cms.untracked.vstring(
                           'l1t', 'yellow',
                           'FwkJob', 
                           'FwkReport', 
                           'FwkSummary', 
                           'Root_NoDictionary'),
    debugModules       = cms.untracked.vstring('*'),

    cout = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    l1t_debug          = cms.untracked.PSet(
       threshold =  cms.untracked.string('DEBUG'),
       default = cms.untracked.PSet (
          limit = cms.untracked.int32(0)
       ),
       l1t = cms.untracked.PSet (
          limit = cms.untracked.int32(100)
       )
    ),
    suppressInfo = cms.untracked.vstring(),
    debugs = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    suppressDebug = cms.untracked.vstring(),
    #cout = cms.untracked.PSet(
    #    placeholder = cms.untracked.bool(True)
    #),
    warnings = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(

    ),
    errors = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    cerr = cms.untracked.PSet(
        optionalPSet = cms.untracked.bool(True),
	INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        FwkSummary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO')
    ),
    FrameworkJobReport = cms.untracked.PSet(
        optionalPSet = cms.untracked.bool(True),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000)
        )
    ),
    suppressWarning = cms.untracked.vstring(),
    statistics = cms.untracked.vstring('cerr_stats'),
    cerr_stats = cms.untracked.PSet(
        optionalPSet = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING'),
        output = cms.untracked.string('cerr')
    ),
    infos = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True),
        optionalPSet = cms.untracked.bool(True),
        Root_NoDictionary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
    ),
)
