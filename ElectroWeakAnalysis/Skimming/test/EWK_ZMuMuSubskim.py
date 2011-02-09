import FWCore.ParameterSet.Config as cms

process = cms.Process("TestZMuMuSubskim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:/scratch2/users/fabozzi/spring10/zmm/38262142-DF46-DF11-8238-0030487C6A90.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('START37_V1A::All')
process.GlobalTag.globaltag = cms.string('MC_3XY_V26::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

############
## no MC truth and on data
process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPaths_cff")

# Output module configuration
process.load("ElectroWeakAnalysis.Skimming.zMuMuSubskimOutputModule_cfi")
process.zMuMuSubskimOutputModule.fileName = 'testZMuMuSubskim.root'

############
# MC truth matching sequence
#process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPathsWithMCTruth_cff")
#process.zMuMuSubskimOutputModule.outputCommands.extend(process.mcEventContent.outputCommands)
############

process.outpath = cms.EndPath(process.zMuMuSubskimOutputModule)


