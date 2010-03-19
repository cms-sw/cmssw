import FWCore.ParameterSet.Config as cms

process = cms.Process("TestZMuMuSubskim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    #'file:../../Skimming/test/mc7Tev.root'
    'rfio:/castor/cern.ch/user/f/fabozzi/mc7tev/F8EE38AF-1EBE-DE11-8D19-00304891F14E.root'
     #'rfio:/castor/cern.ch/user/f/fabozzi/mc7tev/F8EE38AF-1EBE-DE11-8D19-00304891F14E.root'
#    'file:/scratch1/cms/data/summer09/aodsim/zmumu/0016/889E7356-0084-DE11-AF48-001E682F8676.root'
#    'file:testEWKMuSkim.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START3X_V18::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("ElectroWeakAnalysis.ZMuMu.zMuMu_SubskimPathsUserData_cff")

# Output module configuration
process.load("ElectroWeakAnalysis.ZMuMu.zMuMuSubskimOutputModuleUserData_cfi")
process.zMuMuSubskimOutputModule.fileName = 'file:testZMuMuSubskimUserData.root'

process.outpath = cms.EndPath(process.zMuMuSubskimOutputModule)


