import FWCore.ParameterSet.Config as cms

process = cms.Process("eleIso")

process.load("Configuration.EventContent.EventContent_cff")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/22/RelVal-RelValTTbar-1214048167-IDEAL_V2-2nd/0003/1C57E5B2-C040-DD11-AE7A-000423D98804.root')
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
    input = cms.untracked.int32(10)
)

process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff")
process.p1 = cms.Path(process.egammaIsolationSequence)
process.outpath = cms.EndPath(process.out)
