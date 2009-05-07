# simplestMLexample -- an example Configuration file for MessageLogger service:

# This example sets up logging to a file, delivering all messages created by
# LogInfo, LogVerbatim,
# LogWarning, LogPrint,
# LogError, LogProblem, LogImportant
# LogSystem, and LogAbsolute
# but not messages created by LogDebug or LogTrace.
#
# cmsRun simplestMLexample_cfg.py produces simplestML.log.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

# Here is the configuration of the MessgeLogger Service:

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('simplestML'),
    simplestML = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("MLexampleModule_1")

process.p = cms.Path(process.sendSomeMessages)
