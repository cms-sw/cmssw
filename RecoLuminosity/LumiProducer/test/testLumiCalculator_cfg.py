import FWCore.ParameterSet.Config as cms

process = cms.Process("LumiCalculator")
process.load("FWCore.MessageService.MessageLogger_cfi")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
process.maxLuminosityBlocks=cms.untracked.PSet(
    input=cms.untracked.int32(3)
)
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(3)
)
process.source= cms.Source("PoolSource",
#             fileNames=cms.untracked.vstring('/store/relval/CMSSW_3_4_0_pre2/RelValSingleElectronPt10/GEN-SIM-RECO/MC_3XY_V10-v1/0003/BE702AE8-C0BD-DE11-87CA-002618943861.root')
              fileNames=cms.untracked.vstring('rfio:/castor/cern.ch/user/x/xiezhen/MC3XYProcessed.root')
#              firstRun=cms.untracked.uint32(120020),
#              firstLuminosityBlock = cms.untracked.uint32(1),                           
#              firstEvent=cms.untracked.uint32(1),
             )
process.test = cms.EDAnalyzer("LumiCalculator"
             )

process.p1 = cms.Path( process.test )

