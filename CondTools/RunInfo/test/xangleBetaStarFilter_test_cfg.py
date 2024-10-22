import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# define global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '132X_dataRun3_Prompt_v2', '')

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
  fileNames = cms.untracked.vstring("/store/data/Run2023D/AlCaPPSPrompt/ALCARECO/PPSCalMaxTracks-PromptReco-v2/000/370/772/00000/00f29e79-bf03-4a59-b396-46accaa03bfc.root")
)

#process.maxEvents = cms.untracked.PSet(
#  input = cms.untracked.int32(1000)
#)

# filter
process.xangleBetaStarFilter = cms.EDFilter("XangleBetaStarFilter",
  lhcInfoLabel = cms.string(""),
  lhcInfoPerLSLabel = cms.string(""),
  lhcInfoPerFillLabel = cms.string(""),

  useNewLHCInfo = cms.bool(True),

  xangle_min = cms.double(150),
  xangle_max = cms.double(170)
  
)

# plotters
process.plotterBefore = cms.EDAnalyzer("CTPPSLHCInfoPlotter",
  lhcInfoLabel = cms.string(""),
  useNewLHCInfo = cms.bool(True),
  outputFile = cms.string("output_before_filter.root")
)

process.plotterAfter = cms.EDAnalyzer("CTPPSLHCInfoPlotter",
  lhcInfoLabel = cms.string(""),
  useNewLHCInfo = cms.bool(True),
  outputFile = cms.string("output_after_filter.root")
)

# path
process.p = cms.Path(
    process.plotterBefore
    * process.xangleBetaStarFilter
    * process.plotterAfter
)
