import FWCore.ParameterSet.Config as cms

process = cms.Process("TestDimuonReco")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:/scratch1/cms/data/summer09/aodsim/ppMuX/0010/9C519151-5883-DE11-8BC8-001AA0095119.root'
#    'file:/data1/home/fabozzi/cmsrel/skim3_1/CMSSW_3_1_2/src/ElectroWeakAnalysis/Skimming/test/testEWKMuSkim_HLTFilterAndGlobalMuonAndPt10AndEta25.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("ElectroWeakAnalysis.ZReco.dimuons_SkimPaths_cff")

# Output module configuration
process.load("ElectroWeakAnalysis.ZReco.dimuonsOutputModule_cfi")
process.dimuonsOutputModule.fileName = 'file:testSkim_fromOriginal.root'

process.outpath = cms.EndPath(process.dimuonsOutputModule)


