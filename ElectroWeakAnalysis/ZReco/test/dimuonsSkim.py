import FWCore.ParameterSet.Config as cms

process = cms.Process("TestDimuonReco")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/C8CEE598-CB6B-DE11-871F-001D09F2905B.root'
#    'file:/scratch1/users/fabozzi/zmm20_fastsim.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2000) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP31X_V1::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("ElectroWeakAnalysis.ZReco.dimuons_SkimPaths_cff")

## Necessary fixes to run 2.2.X on 2.1.X data
#from PhysicsTools.PatAlgos.tools.cmsswVersionTools import run22XonSummer08AODSIM
#run22XonSummer08AODSIM(process)
#process.source.inputCommands = cms.untracked.vstring(
#        'keep *',
#        'drop *_particleFlow_*_*',
#        #'drop *_particleFlowBlock_*_*',
#)

# Output module configuration
process.load("ElectroWeakAnalysis.ZReco.dimuonsOutputModule_cfi")
process.dimuonsOutputModule.fileName = 'file:/tmp/fabozzi/testSkim_311.root'

process.outpath = cms.EndPath(process.dimuonsOutputModule)


