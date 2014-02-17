import FWCore.ParameterSet.Config as cms
process = cms.Process("ANASKIM")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.GlobalTag.globaltag = 'GR09_R_34X_V5::All'

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    'rfio:/castor/cern.ch/user/e/edwenger/MBSkimHIRECO_900GeV_2k_v3.root'
    #'file:/d100/data/MinimumBias-ReReco/Feb9ReReco_v2/BSCNOHALOSkim/HIRECO/MBSkimHIRECO_900GeV_2k_v3.root'
    ),
    lumisToProcess = cms.untracked.VLuminosityBlockRange(
    '123596:2-123596:max','123615:70-123615:max','123732:62-123732:109',
    '123815:8-123815:max','123818:2-123818:42','123908:2-123908:12',
    '124008:1-124008:1','124009:1-124009:68','124020:12-124020:94',
    '124022:66-124022:179','124023:38-124023:max','124024:2-124024:83',
    '124025:5-124025:13','124027:24-124027:max','124030:2-124030:max')
)

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/HeavyIonsAnalysis/Configuration/test/skimHiCommon_cfg.py,v $'),
    annotation = cms.untracked.string('HI common skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

# =============== Filter Path =====================
process.load("HeavyIonsAnalysis.Configuration.analysisFilters_cff")
process.load("HeavyIonsAnalysis.Configuration.analysisProducers_cff")
process.skimHiCommon = cms.Path(process.bptxAnd *
                                process.bscOr *
                                process.allTracks
                                )


# =============== Output ================================
process.load("HeavyIonsAnalysis.Configuration.analysisEventContent_cff")
process.output = cms.OutputModule("PoolOutputModule",
    process.hiAnalysisSkimContent,
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('skimHiCommon')),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('AOD'),
      filterName = cms.untracked.string('skimHiCommon')),
    fileName = cms.untracked.string('hiCommonSkimAOD.root')
)

process.outpath = cms.EndPath(process.output)

