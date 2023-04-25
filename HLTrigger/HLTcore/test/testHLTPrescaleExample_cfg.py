import FWCore.ParameterSet.Config as cms

## VarParsing
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('analysis')
options.register('globalTag', 'auto:run3_data', options.multiplicity.singleton, options.varType.string, 'name of global tag')
options.setDefault('inputFiles', [
  '/store/data/Run2022D/HLTPhysics/MINIAOD/PromptReco-v2/000/357/898/00000/99afb702-5640-453c-8d48-49b9ba7098d9.root'
])
options.setDefault('maxEvents', 10)
options.parseArguments()

## Process
process = cms.Process('TEST')

process.options.wantSummary = False
process.maxEvents.input = options.maxEvents

## Source
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(options.inputFiles)
)

## MessageLogger
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(5000),
    limit = cms.untracked.int32(10000000)
)

## GlobalTag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

## EventData Modules
_hltPSExample = cms.EDAnalyzer('HLTPrescaleExample',
    hltProcess = cms.string('HLT'),
    hltPath = cms.string(''),
    hltPSProvCfg = cms.PSet(
        stageL1Trigger = cms.uint32(2),
        l1tAlgBlkInputTag = cms.InputTag('gtStage2Digis'),
        l1tExtBlkInputTag = cms.InputTag('gtStage2Digis')
    )
)

process.hltPSExample1 = _hltPSExample.clone(hltPath = 'HLT_Photon33_v6')
process.hltPSExample2 = _hltPSExample.clone(hltPath = 'HLT_Photon50_v14')
process.hltPSExample3 = _hltPSExample.clone(hltPath = 'HLT_Random_v3')

## Path
process.p = cms.Path(
    process.hltPSExample1
  + process.hltPSExample2
  + process.hltPSExample3
)
