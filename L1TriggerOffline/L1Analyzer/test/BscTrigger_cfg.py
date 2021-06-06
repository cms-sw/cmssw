import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load('Configuration/EventContent/EventContent_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_2_2_4/RelValBeamHalo/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/4059446C-1FF3-DD11-9570-001D09F241B4.root',
'/store/relval/CMSSW_2_2_4/RelValBeamHalo/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/60D8AF30-91F3-DD11-81A6-001D09F29597.root'

)
)
process.GlobalTag.globaltag = 'STARTUP_V8::All'
# BSC Trigger simulation 
process.load("L1TriggerOffline.L1Analyzer.bscTrigger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        BscSim = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
            reportEvery = cms.untracked.int32(1)
        ),
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
            reportEvery = cms.untracked.int32(1)
        ),
        FwkSummary = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
            reportEvery = cms.untracked.int32(1)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noTimeStamps = cms.untracked.bool(False),
        threshold = cms.untracked.string('DEBUG'),
        enableStatistics = cms.untracked.bool(True),
        statisticsThreshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring('bscTrigger'),
    default = cms.untracked.PSet(

    ),
    suppressDebug = cms.untracked.vstring(),
    suppressInfo = cms.untracked.vstring(),
    suppressWarning = cms.untracked.vstring()
)
# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3000)
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)




# Event output

process.FEVT = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('BSCTrigger_relval.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('trigger_step')
    )
)

# Path and EndPath definitions

process.trigger_step = cms.Path(process.bscTrigger)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.trigger_step,process.outpath)




