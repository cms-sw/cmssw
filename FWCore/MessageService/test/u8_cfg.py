# Unit test configuration file for MessageLogger service: defaults and limits
# testing overall defaults
#   overall default of no time stamps
#   overall defaults for all destinations, for unnamed categories                     -- u8_overall_unnamed
#   overall defaults for all destinations, specific category                          -- u8_overall_specific
# testing limits
#   default limit for a destination superceding overall default                       -- u8_supercede_specific   
#   default limit for a destination not superceding specific category overall default -- u8_non_supercede_common
#   limit for specific category superceding both defaults                             -- u8_specific

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    default = cms.untracked.PSet(
        expect_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(True),
        expect_overall_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        lim2bycommondefault = cms.untracked.PSet(
            limit = cms.untracked.int32(2)
        ),
        lim0bydefaults = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        expect_supercede_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        expect_non_supercede_common_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        limit = cms.untracked.int32(5),
        expect_overall_unnamed = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    u8_non_supercede_common = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        expect_non_supercede_common_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        noTimeStamps = cms.untracked.bool(True)
    ),
    u8_supercede_specific = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        lim2bycommondefault = cms.untracked.PSet(
            limit = cms.untracked.int32(8)
        ),
        noTimeStamps = cms.untracked.bool(True),
        expect_supercede_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    u8_overall_unnamed = cms.untracked.PSet(
        lim3bydefault = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(True),
        lim2bycommondefault = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        expect_overall_unnamed = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    u8_specific = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        expect_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        noTimeStamps = cms.untracked.bool(True),
        lim2bycommondefault = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        lim0bydefaults = cms.untracked.PSet(
            limit = cms.untracked.int32(6)
        )
    ),
    u8_overall_specific = cms.untracked.PSet(
        lim3bydefault = cms.untracked.PSet(
            limit = cms.untracked.int32(3)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(True),
        expect_overall_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        lim2bycommondefault = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('preEventProcessing', 
        'lim3bydefault', 
        'lim2bycommondefault', 
        'lim0bydefaults', 
        'expect_overall_unnamed', 
        'expect_overall_specific', 
        'expect_supercede_specific', 
        'expect_non_supercede_common_specific', 
        'expect_specific', 
        'FwkTest'),
    destinations = cms.untracked.vstring('u8_overall_unnamed', 
        'u8_overall_specific', 
        'u8_supercede_specific', 
        'u8_non_supercede_common', 
        'u8_specific')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_E")

process.p = cms.Path(process.sendSomeMessages)
