import FWCore.ParameterSet.Config as cms

process = cms.Process("TestZMuMuSubskim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
#    'file:/scratch1/cms/data/summer09/aodsim/zmumu/0016/889E7356-0084-DE11-AF48-001E682F8676.root'
    'file:testEWKMuSkim.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPaths_cff")

# Output module configuration
process.load("ElectroWeakAnalysis.Skimming.zMuMuSubskimOutputModule_cfi")
process.zMuMuSubskimOutputModule.fileName = 'file:testZMuMuSubskim.root'

process.outpath = cms.EndPath(process.zMuMuSubskimOutputModule)


