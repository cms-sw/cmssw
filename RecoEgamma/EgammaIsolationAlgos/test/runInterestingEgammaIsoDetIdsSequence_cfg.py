import FWCore.ParameterSet.Config as cms

process = cms.Process("egamIsoDetIds")

process.load("Configuration.StandardSequences.Geometry_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('REPLACE ME')
)

process.load("Configuration.EventContent.EventContent_cff")
process.out = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('file:egammaDetIds.root')
)

process.out.outputCommands.append('drop *_*_*_*')
process.out.outputCommands.append('keep *_*_*_egamIsoDetIds')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#process.Timing = cms.Service("Timing")

process.load("RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoDetIdsSequence_cff")
process.p1 = cms.Path(process.interestingEgammaIsoDetIds)
process.outpath = cms.EndPath(process.out)
