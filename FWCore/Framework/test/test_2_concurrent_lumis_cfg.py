import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource", numberEventsInLuminosityBlock = cms.untracked.uint32(2))

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32( 20 ) )

process.options = cms.untracked.PSet( numberOfThreads = cms.untracked.uint32(4),
                                      numberOfStreams = cms.untracked.uint32(0),
                                      numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(2))

process.prod = cms.EDProducer("BusyWaitIntProducer",
                              ivalue = cms.int32(1),
                              iterations = cms.uint32(50*1000) )

process.p = cms.Path(process.prod)

process.add_(cms.Service("ConcurrentModuleTimer",
                         modulesToExclude = cms.untracked.vstring("TriggerResults","p"),
                         excludeSource = cms.untracked.bool(True)))

