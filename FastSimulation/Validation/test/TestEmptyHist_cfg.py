import FWCore.ParameterSet.Config as cms

process = cms.Process("OWNPARTICLES")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("EmptySource")

process.load("FastSimulation.Validation.EmptySimHits_cfi")
process.emptySimHits.pCaloHitInstanceLabels = cms.vstring("CastorFI")
process.emptySimHits.pSimHitInstanceLabels = cms.vstring("muonCSCHits","muonRPCHits")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root')
)

process.p = cms.Path(process.emptySimHits)

process.e = cms.EndPath(process.out)
