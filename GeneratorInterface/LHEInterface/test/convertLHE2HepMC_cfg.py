import FWCore.ParameterSet.Config as cms

process = cms.Process("convertLHE2HepMC")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:gen.root')
)

process.load("GeneratorInterface.LHEInterface.lhe2HepMCConverter_cfi")

process.out = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('out.root'),
    outputCommands = cms.untracked.vstring('drop *','keep *_lhe2HepMCConverter_*_*'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('p'),
        dataTier = cms.untracked.string('GEN')
    )
)


process.p = cms.Path(process.lhe2HepMCConverter)
process.e = cms.EndPath(process.out)
