# Unit test configuration file for MessageLogger service: reportEverys
# testing overall defaults regarding reportEverys
#   overall defaults for all destinations, for unnamed categories                     -- u11_overall_unnamed
#   overall defaults for all destinations, specific category                          -- u11_overall_specific
# testing reportEverys
#   default reportEvery for a destination superceding overall default                       -- u11_supercede_specific   
#   default reportEvery for a destination not superceding specific category overall default -- u11_non_supercede_common
#   reportEvery for specific category superceding both defaults                             -- u11_specific
# In the course of this, these cases also test reportEverys and limits interacting with each other

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    default = cms.untracked.PSet(
        expect_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(True),
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        reportEvery = cms.untracked.int32(5),
        expect_supercede_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        int7bycommondefault = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(7),
            limit = cms.untracked.int32(-1)
        ),
        expect_non_supercede_common_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        int25bydefaults = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(25),
            limit = cms.untracked.int32(-1)
        ),
        expect_overall_unnamed = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        expect_overall_specific = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    files = cms.untracked.PSet(
        u11_supercede_specific = cms.untracked.PSet(
            default = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            int4bydefault = cms.untracked.PSet(
                reportEvery = cms.untracked.int32(10),
                limit = cms.untracked.int32(-1)
            ),
            expect_supercede_specific = cms.untracked.PSet(
                reportEvery = cms.untracked.int32(1),
                limit = cms.untracked.int32(-1)
            ),
            int25bydefaults = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            int7bycommondefault = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            )
        ),
        u11_overall_specific = cms.untracked.PSet(
            default = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            int4bydefault = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            expect_overall_specific = cms.untracked.PSet(
                reportEvery = cms.untracked.int32(1),
                limit = cms.untracked.int32(-1)
            ),
            int25bydefaults = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            )
        ),
        u11_non_supercede_common = cms.untracked.PSet(
            int4bydefault = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            expect_non_supercede_common_specific = cms.untracked.PSet(
                reportEvery = cms.untracked.int32(1),
                limit = cms.untracked.int32(-1)
            ),
            default = cms.untracked.PSet(
                reportEvery = cms.untracked.int32(18),
                limit = cms.untracked.int32(3)
            ),
            int25bydefaults = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            )
        ),
        u11_specific = cms.untracked.PSet(
            int4bydefault = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            expect_specific = cms.untracked.PSet(
                reportEvery = cms.untracked.int32(1),
                limit = cms.untracked.int32(-1)
            ),
            noTimeStamps = cms.untracked.bool(True),
            default = cms.untracked.PSet(
                reportEvery = cms.untracked.int32(30)
            ),
            int25bydefaults = cms.untracked.PSet(
                reportEvery = cms.untracked.int32(12)
            ),
            int7bycommondefault = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            )
        ),
        u11_overall_unnamed = cms.untracked.PSet(
            int4bydefault = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            int25bydefaults = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            int7bycommondefault = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            expect_overall_unnamed = cms.untracked.PSet(
                reportEvery = cms.untracked.int32(1),
                limit = cms.untracked.int32(-1)
            )
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_F")

process.p = cms.Path(process.sendSomeMessages)
