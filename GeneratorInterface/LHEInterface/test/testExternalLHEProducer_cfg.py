import FWCore.ParameterSet.Config as cms

process = cms.Process("ScriptExample")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource",
    firstLuminosityBlock = cms.untracked.uint32(23456)                            
)                            

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.load("GeneratorInterface/LHEInterface/ExternalLHEProducer_cfi")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root')
)

process.p = cms.Path(process.externalLHEProducer)

process.e = cms.EndPath(process.out)
