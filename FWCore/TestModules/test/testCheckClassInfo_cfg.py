import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# load the latest HLT configuration in order to exercise a large number of event products
process.load('HLTrigger.Configuration.HLT_GRun_cff')

process.options.numberOfThreads = 8
process.options.numberOfStreams = 8

# do not process any events, only construct the modules
process.source = cms.Source("EmptySource")
process.maxEvents.input = 0

# show the CheckClassInfo messages
process.MessageLogger.CheckClassInfo = cms.untracked.PSet()

# check the ROOT dictionaries of all non-transient products declared by any module 
process.checkAll = cms.EDAnalyzer("edmtest::CheckClassInfo",
    eventProducts = cms.untracked.vstring("*")
)

process.CheckClassInfo = cms.Path(
    process.checkAll
)

process.schedule.append( process.CheckClassInfo )
