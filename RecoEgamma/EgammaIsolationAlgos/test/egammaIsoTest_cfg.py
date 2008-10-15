import FWCore.ParameterSet.Config as cms

process = cms.Process("eleIso")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.EventContent.EventContent_cff")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('FILENAME')
)

process.out = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('file:eleIso.root')
)

process.out.outputCommands.append('drop *_*_*_*')
process.out.outputCommands.append('keep *_pixelMatchGsfElectrons_*_*')
process.out.outputCommands.append('keep *_photons_*_*')
process.out.outputCommands.append('keep *_*_*_eleIso')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(4000)
)

process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff")
process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequencePAT_cff")
process.p1 = cms.Path(process.egammaIsolationSequence + process.egammaIsolationSequencePAT)
process.outpath = cms.EndPath(process.out)
