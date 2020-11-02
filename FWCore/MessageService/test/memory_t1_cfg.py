# memory_t1.cfg
# EXTERNAL Unit test configuration file for Memory service:
# output to JobReport

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(False),
    showMallocInfo = cms.untracked.bool(False),
    ignoreTotal = cms.untracked.int32(5)
)

process.MessageLogger = cms.Service("MessageLogger",
    o1_infos = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    debugModules = cms.untracked.vstring('*'),
    categories = cms.untracked.vstring('preEventProcessing', 
        'FwkReport', 
        'FwkTest'),
    destinations = cms.untracked.vstring('memory_t1_infos')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(35)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("MemoryTestClient_A",
    pattern = cms.untracked.int32(2)
)

process.p = cms.Path(process.sendSomeMessages)
