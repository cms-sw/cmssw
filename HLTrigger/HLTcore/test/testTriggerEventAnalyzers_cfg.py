import FWCore.ParameterSet.Config as cms

## VarParsing
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('logLevel', 'WARNING', options.multiplicity.singleton, options.varType.string, 'value of MessageLogger.cerr.threshold')
options.register('globalTag', '125X_mcRun3_2022_realistic_v3', options.multiplicity.singleton, options.varType.string, 'name of GlobalTag')
options.setDefault('inputFiles', [
  '/store/relval/CMSSW_12_6_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/125X_mcRun3_2022_realistic_v3-v1/2580000/2d96539c-b321-401f-b7b2-51884a5d421f.root',
])
options.setDefault('maxEvents', 10)
options.parseArguments()

## Process
process = cms.Process('TEST')

process.options.numberOfThreads = 1
process.options.numberOfStreams = 0
process.options.wantSummary = False
process.maxEvents.input = options.maxEvents

## Source
process.source = cms.Source('PoolSource',
  fileNames = cms.untracked.vstring(options.inputFiles)
)

## GlobalTag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

## EventData Modules
from HLTrigger.HLTcore.hltEventAnalyzerAOD_cfi import hltEventAnalyzerAOD as _hltEventAnalyzerAOD
process.triggerEventAnalyzer = _hltEventAnalyzerAOD.clone(
  processName = 'HLT',
  triggerName = '@',
  triggerResults = 'TriggerResults::HLT',
  triggerEvent = 'hltTriggerSummaryAOD::HLT',
  stageL1Trigger = 2,
  verbose = False,
)

from HLTrigger.HLTcore.hltEventAnalyzerRAW_cfi import hltEventAnalyzerRAW as _hltEventAnalyzerRAW
process.triggerEventWithRefsAnalyzer = _hltEventAnalyzerRAW.clone(
  processName = 'HLT',
  triggerName = '@',
  triggerResults = 'TriggerResults::HLT',
  triggerEventWithRefs = 'hltTriggerSummaryRAW::HLT',
  verbose = False,
  permissive = True,
)

from HLTrigger.HLTcore.triggerSummaryAnalyzerAOD_cfi import triggerSummaryAnalyzerAOD as _triggerSummaryAnalyzerAOD
process.triggerEventSummaryAnalyzer = _triggerSummaryAnalyzerAOD.clone(
  inputTag = 'hltTriggerSummaryAOD'
)

from HLTrigger.HLTcore.triggerSummaryAnalyzerRAW_cfi import triggerSummaryAnalyzerRAW as _triggerSummaryAnalyzerRAW
process.triggerEventWithRefsSummaryAnalyzer = _triggerSummaryAnalyzerRAW.clone(
  inputTag = 'hltTriggerSummaryRAW'
)

## MessageLogger
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1 # only report every Nth event start
process.MessageLogger.cerr.FwkReport.limit = -1      # max number of reported messages (all if -1)
process.MessageLogger.cerr.enableStatistics = False  # enable "MessageLogger Summary" message
process.MessageLogger.cerr.threshold = options.logLevel
setattr(process.MessageLogger.cerr, options.logLevel,
  cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1), # every event!
    limit = cms.untracked.int32(-1)       # no limit! (default is limit=0, i.e. no messages reported)
  )
)
process.MessageLogger.HLTEventAnalyzerAOD = dict()
process.MessageLogger.HLTEventAnalyzerRAW = dict()
process.MessageLogger.TriggerSummaryAnalyzerAOD = dict()
process.MessageLogger.TriggerSummaryAnalyzerRAW = dict()

## Path
process.triggerEventAnalysisPath = cms.Path(
    process.triggerEventAnalyzer
  + process.triggerEventSummaryAnalyzer
  + process.triggerEventWithRefsAnalyzer
  + process.triggerEventWithRefsSummaryAnalyzer
)
