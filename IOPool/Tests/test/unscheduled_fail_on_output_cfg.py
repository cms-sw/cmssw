import FWCore.ParameterSet.Config as cms

process = cms.Process('Test')

process.source = cms.Source('EmptySource')
process.failing = cms.EDProducer('FailingProducer')
process.i = cms.EDProducer('IntProducer',
                           ivalue = cms.int32(10) )
process.out = cms.OutputModule('PoolOutputModule',
                                fileName = cms.untracked.string('unscheduled_fail_on_output.root'))

process.t = cms.Task(process.failing)
process.o = cms.EndPath(process.out, process.t)
process.p = cms.Path(process.i)

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10)
        )
