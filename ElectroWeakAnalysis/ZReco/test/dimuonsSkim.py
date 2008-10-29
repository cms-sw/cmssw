import FWCore.ParameterSet.Config as cms

process = cms.Process("TestDimuonReco")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_V9::All'
process.load("Configuration.StandardSequences.MagneticField_cff")


process.load("ElectroWeakAnalysis.ZReco.dimuons_SkimPaths_cff")
process.load("ElectroWeakAnalysis.ZReco.dimuonsOutputModule_cfi")

process.dimuonsOutputModule.fileName = 'file:/tmp/fabozzi/dimuons.root'

process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)
                                                  
process.maxEvents = cms.untracked.PSet(
  input =cms.untracked.int32(100)
)

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
       'file:/tmp/fabozzi/06029757-B588-DD11-BDD7-001CC4AA8E08.root'
  )
)

process.endp = cms.EndPath(
  process.dimuonsOutputModule
)


