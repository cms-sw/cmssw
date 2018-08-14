import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.intMaker = cms.EDProducer("edmtest::FailingInRunProducer")

process.consumer = cms.EDAnalyzer("edmtest::IntFromRunConsumingAnalyzer",
                                  getFromModule = cms.untracked.InputTag("intMaker"))

process.p = cms.Path(process.consumer,cms.Task(process.intMaker))

process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(2),
                                     numberOfStreams = cms.untracked.uint32(1))
