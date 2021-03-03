# memory_t3.cfg
# EXTERNAL Unit test configuration file for Memory service:
# Module output to JobReport
# output to JobReport

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(False),
    showMallocInfo = cms.untracked.bool(False),
    ignoreTotal = cms.untracked.int32(10)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        memory_t3_infos = cms.untracked.PSet(

        )
    ),
    o1_infos = cms.untracked.PSet(
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(True),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('INFO')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(40)
)

process.source = cms.Source("EmptySource")

process.module1 = cms.EDAnalyzer("MemoryTestClient_B",
    pattern = cms.untracked.int32(1)
)

process.module2 = cms.EDAnalyzer("MemoryTestClient_B",
    pattern = cms.untracked.int32(2)
)

process.module3 = cms.EDAnalyzer("MemoryTestClient_B",
    pattern = cms.untracked.int32(3)
)

process.module4 = cms.EDAnalyzer("MemoryTestClient_B",
    pattern = cms.untracked.int32(4)
)

process.p = cms.Path(process.module1*process.module2*process.module3*process.module4)
