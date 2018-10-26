import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.MessageLogger.cerr.INFO = cms.untracked.PSet(
#    reportEvery = cms.untracked.int32(1), # every!
#    limit = cms.untracked.int32(-1)       # no limit!
#    )
#process.MessageLogger.cerr.FwkReport.reportEvery = 10 # only report every 10th event start
#process.MessageLogger.cerr_stats.threshold = 'INFO' # also info in statistics

# load conditions from the global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# read back the trigger decisions
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:trigger.root')
)

from HLTrigger.HLTfilters.eventBXFilter_cfi import eventBXFilter
process.empty = eventBXFilter.clone(
   allowedBXs = cms.vuint32(),
   vetoBXs    = cms.vuint32()
)
process.allowedBX = eventBXFilter.clone(
   allowedBXs = cms.vuint32(536),
   vetoBXs    = cms.vuint32()
)
process.allowedBX12 = eventBXFilter.clone(
   allowedBXs = cms.vuint32(11,536),
   vetoBXs    = cms.vuint32()
)
process.vetoBX1 = eventBXFilter.clone(
   allowedBXs = cms.vuint32(),
   vetoBXs    = cms.vuint32(1)
)
process.vetoBX = eventBXFilter.clone(
   allowedBXs = cms.vuint32(),
   vetoBXs    = cms.vuint32(536)
)
process.vetoBX12 = eventBXFilter.clone(
   allowedBXs = cms.vuint32(),
   vetoBXs    = cms.vuint32(11,536)
)
process.warning = eventBXFilter.clone(
   allowedBXs = cms.vuint32(1),
   vetoBXs    = cms.vuint32(1)
)


process.path_empty        = cms.Path( process.empty )
process.path_allowedBX    = cms.Path( process.allowedBX )
process.path_allowedBX12  = cms.Path( process.allowedBX12 )
process.path_vetoBX1      = cms.Path( process.vetoBX1 )
process.path_vetoBX       = cms.Path( process.vetoBX )
process.path_vetoBX12     = cms.Path( process.vetoBX12 )
process.path_warning      = cms.Path( process.warning )


# define an EndPath to analyze all other path results
process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
    HLTriggerResults = cms.InputTag( 'TriggerResults','','TEST' )
)
process.HLTAnalyzerEndpath = cms.EndPath( process.hltTrigReport )
