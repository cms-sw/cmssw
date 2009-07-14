import FWCore.ParameterSet.Config as cms

process = cms.Process("TestDimuonReco")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:/scratch1/cms/data/summer08/zmumu/06029757-B588-DD11-BDD7-001CC4AA8E08.root'
#    'file:/scratch1/users/fabozzi/zmm20_fastsim.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('IDEAL_V9::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("ElectroWeakAnalysis.ZReco.dimuons_SkimPaths_cff")

## Necessary fixes to run 2.2.X on 2.1.X data
#from PhysicsTools.PatAlgos.tools.cmsswVersionTools import run22XonSummer08AODSIM
#run22XonSummer08AODSIM(process)
process.source.inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_particleFlow_*_*',
        #'drop *_particleFlowBlock_*_*',
)

# Output module configuration
process.load("ElectroWeakAnalysis.ZReco.dimuonsOutputModule_cfi")
process.dimuonsOutputModule.fileName = 'file:testSkim_triggermatch.root'

process.outpath = cms.EndPath(process.dimuonsOutputModule)


