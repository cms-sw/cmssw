# Test of a feature of PSet validation:
#   Category nested PSets

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",

    destinations = cms.untracked.vstring( 'u1_warnings',  'u1_errors',
                   'u1_infos',  'u1_debugs', 'u1_default', 'u1_x'), 
    statistics = cms.untracked.vstring( 'u1_warnings', 'u1_default', 'u1_y' ), 
    categories = cms.untracked.vstring('preEventProcessing','FwkTest',
                                       'cat_A','cat_B', 'cat_J', 'cat_K'),

# enabling any of these acter the first one should fail:

    cat_J =  cms.untracked.PSet(
            limit = cms.untracked.int32(100),
            reportEvery = cms.untracked.int32(10),
	    timespan =  cms.untracked.int32(10)
        ),

#    cat_K =  cms.untracked.PSet(
#            limit = cms.untracked.int32(100),
#            reportEvery = cms.untracked.int32(10),
#	    timespan =  cms.untracked.int32(10),
#	    nonsense = cms.untracked.int32(10)
#        ),
 
# enabling any of these except the first 5 should fail:

    u1_x = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
#    cat_K =  cms.untracked.PSet(
#            limit = cms.untracked.int32(100),
#            reportEvery = cms.untracked.int32(10),
#	    timespan =  cms.untracked.int32(10),
#	    nonsense = cms.untracked.int32(10)
#        ),

#    cat_J =  cms.untracked.PSet(
#            a  = cms.untracked.bool(True),
#	     b = cms.untracked.PSet( c = cms.untracked.int32(10) )
#        ),
   	
# this one should be fine
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),

   
    u1_infos = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        FwkTest = cms.untracked.PSet(
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
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    u1_default = cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(True),
        FwkTest = cms.untracked.PSet(
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
