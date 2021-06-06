# timingFilteredMLexample -- 
#                  an example Configuration file for MessageLogger service:

# Illustrates how to limit the output of a category of message to one
# destination but not to another.
#
# This example sets up logging to a file and to cout, and establishes the
# timing service.  The idea is to deliver all messages created by LogWarning 
# and above to cout (as would happen in usual production jobs) EXCEPT to
# suppress the timing messages for cout (but not for the other log file).
#
# This sort of setup was requested by Florian Beaudette, Jean-Roch, Peter Elmer
# and for a time was rendered impossible because the timing service had used
# LogAbsolute for its output; that has since been modified.
#
# cmsRun timingFilteredMLexample_cfg.py outputs to cout, and also produces 
# timingFilteredMLexample.log.  The output to cout should not contain per-event
# timing information.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

# Here is where the timng service is enabled

process.Timing =  cms.Service("Timing")

# Here is the configuration of the MessgeLogger Service:

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        TimeEvent = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TimeModule = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TimeReport = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        cat_B = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    ),
    files = cms.untracked.PSet(
        timingFilteredML = cms.untracked.PSet(
            threshold = cms.untracked.string('WARNING')
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("MLexampleModule_1")

process.p = cms.Path(process.sendSomeMessages)
