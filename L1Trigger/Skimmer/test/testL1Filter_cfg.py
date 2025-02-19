import FWCore.ParameterSet.Config as cms

process = cms.Process("L1SKIM")
# initialize  MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("rfio:/castor/cern.ch/user/e/emiglior/Alignment/SkimmedData/TestAlCaRecoWithGT_50911_10k.root")
)

process.load("L1Trigger.Skimmer.l1Filter_cfi")

process.filterPath = cms.Path(process.l1Filter)
