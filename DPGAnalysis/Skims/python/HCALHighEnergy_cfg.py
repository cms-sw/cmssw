import FWCore.ParameterSet.Config as cms
import DPGAnalysis.Skims.HCALHighEnergyCombinedPath_cff

process = cms.Process("USER")

process.extend(DPGAnalysis.Skims.HCALHighEnergyCombinedPath_cff)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
       '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/122/D424EBA5-55A0-DD11-A8BF-000423D9853C.root',
       '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/122/C67EDF0D-49A0-DD11-9403-001617DBD332.root')
)                            

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/HCALHighEnergy_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT HCALHighEnergy skim')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *'),
                               SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring("HCALHighEnergyPath")
    ),
                               dataset = cms.untracked.PSet(
                               dataTier = cms.untracked.string('RECO'),
                               filterName = cms.untracked.string('HCALHighEnergy')),
                               fileName = cms.untracked.string('HCALHighEnergy_filter.root')
                               )

process.p = cms.EndPath(process.out)
