import FWCore.ParameterSet.Config as cms

process = cms.Process('JustATest')

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.Geometry.GeometryIdeal_cff")

process.load('RecoBTag/Configuration/RecoBTag_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

## Options and Output Report
process.options   = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
      '/store/relval/CMSSW_7_6_0_pre3/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/2E84FB77-FC41-E511-9B44-0025905A612C.root',
      '/store/relval/CMSSW_7_6_0_pre3/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/3A7D247C-FC41-E511-BEE0-002618943981.root',
      ),
)

process.out = cms.OutputModule(
   "PoolOutputModule",
   fileName=cms.untracked.string("ctag_test.root"),
   outputCommands=cms.untracked.vstring("keep *")
)

process.p = cms.Path(
    process.pfCTagging
)

process.end = cms.EndPath(
   process.out
)
