# Unit test configuration file for MessageLogger service:
# threshold levels for destinations
# limit=0 for a category (needed to avoid time stamps in files to be compared)
# enabling all (*) LogDebug, with one destination responding
# verify that by default, the threshold for a destination is INFO
# also verify name used for "severe" errors is System, not Severe, in summary

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring('u17_all'),
    categories = cms.untracked.vstring('cat_P', 
        'cat_S', 
        'FwkTest',
	'FwkReport'),
    u17_all = cms.untracked.PSet(
	threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(110)
        ),
        cat_P = cms.untracked.PSet(
            limit = cms.untracked.int32(2)
        ),
        cat_S = cms.untracked.PSet(
            limit = cms.untracked.int32(2)
        )
    ),
    destinations = cms.untracked.vstring('u17_all')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_K")

process.p = cms.Path(process.sendSomeMessages)
