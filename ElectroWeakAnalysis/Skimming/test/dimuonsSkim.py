import FWCore.ParameterSet.Config as cms

process = cms.Process("TestDimuonReco")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:/scratch1/cms/data/relval_382/zmm/62C86D62-BFAF-DF11-85B3-003048678A6C.root'
  )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START38_V9::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("ElectroWeakAnalysis.Skimming.dimuons_SkimPaths_cff")

# Output module configuration
process.load("ElectroWeakAnalysis.Skimming.dimuonsOutputModule_cfi")
process.dimuonsOutputModule.fileName = 'file:testDimuonSkim.root'

process.outpath = cms.EndPath(process.dimuonsOutputModule)


