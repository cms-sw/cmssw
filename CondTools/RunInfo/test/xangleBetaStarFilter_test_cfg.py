import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# define global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '106X_dataRun2_v28', '')

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    )
)

# data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring("/store/data/Run2018D/EGamma/MINIAOD/12Nov2019_UL2018-v4/280000/FF9D0498-30CA-7241-A85C-6F4F272A7A16.root")
)

#process.maxEvents = cms.untracked.PSet(
#  input = cms.untracked.int32(1000)
#)

# filter
process.load("CondTools.RunInfo.xangleBetaStarFilter_cfi")
process.xangleBetaStarFilter.xangle_min = 150
process.xangleBetaStarFilter.xangle_max = 170

# plotters
process.plotterBefore = cms.EDAnalyzer("CTPPSLHCInfoPlotter",
  lhcInfoLabel = cms.string(""),
  outputFile = cms.string("output_before_filter.root")
)

process.plotterAfter = cms.EDAnalyzer("CTPPSLHCInfoPlotter",
  lhcInfoLabel = cms.string(""),
  outputFile = cms.string("output_after_filter.root")
)

# path
process.p = cms.Path(
    process.plotterBefore
    * process.xangleBetaStarFilter
    * process.plotterAfter
)
