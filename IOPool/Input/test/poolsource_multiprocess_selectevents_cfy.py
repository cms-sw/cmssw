# Prepare a file that has empty Runs and Lumis
# We will read it in the next step
#
# Here is the content of the output data file produced by
# this configuration:
#
# There are 20 runs numbered consecutively from 100 to 119
# Originally there were 5 events in every run with
# lumi 1 containing events 1 and 2
# lumi 2 containing events 3 and 4
# lumi 3 containing event 5
# This configuration selects only some of the events so in the
# output this is the distribution of events.
#
# 100-101 empty runs
# 102 has 5 events
# 103 empty run
# 104 has 5 events
# 105 has event 1 and 5 (middle lumi is empty)
# 106 has event 3 (first and last lumi's are empty)
# 107 has 5 events
# 108-109 empty runs
# 110 has events 1 to 3 (last lumi empty)
# 111 empty run
# 112 has events 3 to 5 (first lumi empty)
# 113-119 are all empty runs

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:multiprocess_test.root")
                            , eventsToProcess = cms.untracked.VEventRange('102:1:1-102:3:5',
                                                                          '104:1:1-104:3:5',
                                                                          '105:1:1-105:1:1',
                                                                          '105:3:5-105:3:5',
                                                                          '106:2:3-106:2:3',
                                                                          '107:1:1-107:3:5',
                                                                          '110:1:1-110:2:3',
                                                                          '112:2:3-112:3:5')
)

process.out = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("multiprocess_test2.root"))

process.p = cms.EndPath(process.out)
