import FWCore.ParameterSet.Config as cms

process = cms.Process("TestEHW")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:in.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)
process.sols = cms.EDFilter("TestEventHypothesisWriter",
    jets = cms.InputTag("iterativeCone5CaloJets"),
    muons = cms.InputTag("pixelMatchGsfElectrons")
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_pixelMatchGsfElectrons_*_*', 
        'keep *_*_*_TestEHW'),
    fileName = cms.untracked.string('eh.root')
)

process.p = cms.Path(process.sols)
process.e = cms.EndPath(process.out)


