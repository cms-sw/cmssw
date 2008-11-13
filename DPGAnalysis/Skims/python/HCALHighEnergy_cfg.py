import FWCore.ParameterSet.Config as cms
import DPGAnalysis.Skims.HCALHighEnergyCombinedPath_cff

process = cms.Process("USER")

process.extend(DPGAnalysis.Skims.HCALHighEnergyCombinedPath_cff)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/data/Commissioning08/Calo/RECO/v1/000/069/365/4E2DE85F-25AB-DD11-8722-001617C3B70E.root'
    )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
    )

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *'),
                               SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring("HCALHighEnergyPath")
    ),
                               fileName = cms.untracked.string('HCALHighEnergy_filter.root')
                               )

process.p = cms.EndPath(process.out)
