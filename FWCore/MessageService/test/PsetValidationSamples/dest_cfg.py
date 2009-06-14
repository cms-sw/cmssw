# Test of a feature of PSet validation:
#   A destination PSet

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",

    destinations = cms.untracked.vstring( 'u1_warnings',  'u1_errors',
                   'u1_infos',  'u1_debugs', 'u1_default', 'u1_x'), 
    statistics = cms.untracked.vstring( 'u1_warnings', 'u1_default', 'u1_y' ), 
    fwkJobReports = cms.untracked.vstring( 'u1_f' ), 
    categories = cms.untracked.vstring('preEventProcessing','FwkJob',
                                       'cat_A','cat_B'),
 
# enabling any of these except the first 5 should fail:

    u1_x = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
#	placeholder = cms.untracked.int32(2),
#	limit = cms.untracked.int32(2),
#	noTimeStamps = cms.untracked.int32(2),
#        cat_C = cms.untracked.PSet(
#          limit = cms.untracked.int32(0)
#        ),
#	filename = cms.untracked.string("cerr"),
#	extension = cms.untracked.string("cout"),
# this one should be fine
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),

# enabling this should lead to error
#    u1_y = cms.untracked.PSet(
#        noTimeStamps = cms.untracked.bool(True)
#    ),

   
    u1_infos = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    u1_warnings = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noTimeStamps = cms.untracked.bool(True)
    ),
    u1_debugs = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        noTimeStamps = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    u1_default = cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    u1_errors = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        noTimeStamps = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring('*'),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)
