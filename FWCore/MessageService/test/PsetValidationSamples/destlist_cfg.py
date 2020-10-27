# Test of a feature of PSet validation:
#   The vstring destinations list

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",

#enable one of the following -- the first should pass, the rest fail
    destinations = cms.untracked.vstring( 'u1_warnings',  'u1_errors',
                   'u1_infos',  'u1_debugs', 'u1_default', 'u1_x'), 
        
#    destinations = cms.untracked.vstring('cerr'),
#    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.vstring('cout'),
#    destinations = cms.untracked.int32(2),
#    destinations = cms.untracked.vstring('u1_warnings', 'u1_errors', 'u1_warnings'),
#    destinations = cms.untracked.vstring('cerr', 'cout'),
#    destinations = cms.untracked.vstring('cout','limit'),

#enable first destinations and one of these -- should fail
#   u1_x = cms.untracked.int32(0),
#   u1_x = cms.untracked.bool(true),
#   u1_x = cms.untracked.string('abc'),
#  u1_x = cms.untracked.vstring('abc','def'),
#   u1_x = cms.int32(0),
#   u1_x = cms.bool(true),
#   u1_x = cms.string('abc'),
#   u1_x = cms.vstring('abc','def'),
    
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
    categories = cms.untracked.vstring('preEventProcessing', 
        'FwkTest')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)
