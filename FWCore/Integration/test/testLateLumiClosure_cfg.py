# This test originally demonstrated a problem
# reported online. The problem has since been
# fixed by a modification in the function
# EventProcessor::handleNextEventForStreamAsync.
# I've left unmodified a description of the
# problem below. It is still interesting to keep
# this configuration around and running as it
# still demonstrates the circumstances in a
# way that none of the other tests replicate
# and shows that the problem is fixed if one
# examines the log output. Note that it is not
# coded as a pass/fail test because the timing
# could vary and we don't want a test that
# occasionally fails. It might sometimes still
# exhibit the problem behavior if for example
# a thread hangs.

# Demonstrates a problem first noticed online. Say there
# are 3 lumis. The first and last lumis have events.
# The middle one has no events. We are using
# a source like the one used online that will wait
# to return from getNextItemType until an event
# is available and may sleep for some period
# of time. The problem is that the first lumi
# does not close at the point where it could
# close but is stuck open until lumi 3 is encountered.
# Between encountering lumi 2 and lumi 3, a
# task necessary to complete lumi 1 is stuck
# in the source serial queue behind the task
# waiting for getNextItemType to return.

# Critical to emulating the problem below is
# having event 3 take a long enough time to
# process that it completes after event 4 has
# completed and after the stream it is on has
# completed stream end lumi and its stream is
# waiting for getNextItemType to return what
# to do next. It is blocking the stream serial
# queue which blocks the stream that processed
# event 3 from running the end lumi stream
# transition.

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.options = dict(
    numberOfThreads = 2,
    numberOfStreams = 2,
    numberOfConcurrentRuns = 1,
    numberOfConcurrentLuminosityBlocks = 2
)

process.Tracer = cms.Service("Tracer",
    printTimestamps = cms.untracked.bool(True)
)

process.source = cms.Source("SourceWithWaits",
    timePerLumi = cms.untracked.double(1),
    sleepAfterStartOfRun = cms.untracked.double(0.25),
    eventsPerLumi = cms.untracked.vuint32(4,0,5),
    lumisPerRun = cms.untracked.uint32(100)
)

process.sleepingProducer = cms.EDProducer("timestudy::SleepingProducer",
    ivalue = cms.int32(1),
    consumes = cms.VInputTag(),
    eventTimes = cms.vdouble(0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
)

process.p = cms.Path(process.sleepingProducer)
