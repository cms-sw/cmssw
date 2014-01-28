import FWCore.ParameterSet.Config as cms

process = cms.Process("testsynch")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/GlobalNov07/A/000/030/205/RAW/0000/383D0C16-A1A2-DC11-9728-000423D60BF6.root'
    )
)

process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
es_prefer_GlobalTag = cms.ESPrefer("PoolDBESSource","GlobalTag")
process.GlobalTag.globaltag = 'CRUZET4_V5P::All'
process.GlobalTag.connect = 'frontier://FrontierProd/CMS_COND_21X_GLOBALTAG'

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("DQM.L1TMonitor.BxTiming_cfi")

#process.bxTiming.VerboseFlag = 1
process.bxTiming.HistFile = 'l1timing.root'
#process.bxTiming.RunInFilterFarm=True

process.p = cms.Path( cms.SequencePlaceholder("RawToDigi") * process.bxTiming )

process.outputEvents = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('testsynch.root')
)


