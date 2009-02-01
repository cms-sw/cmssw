import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load('Configuration/EventContent/EventContent_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'rfio:/castor/cern.ch/user/b/boudoul/220/BH/BSCTrigger/Reco/step2_TrackerHaloMuon_1.root',
'rfio:/castor/cern.ch/user/b/boudoul/220/BH/BSCTrigger/Reco/step2_TrackerHaloMuon_2.root',
'rfio:/castor/cern.ch/user/b/boudoul/220/BH/BSCTrigger/Reco/step2_TrackerHaloMuon_3.root'

)
)
process.GlobalTag.globaltag = 'STARTUP_V7::All'
# BSC Trigger simulation 
process.load("L1TriggerOffline.L1Analyzer.bscTrigger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    suppressInfo = cms.untracked.vstring(),
    debugs = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    suppressDebug = cms.untracked.vstring(),
    cout = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    warnings = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(

    ),
    errors = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    cerr = cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(False),
        BscSim = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        FwkReport = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkSummary = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('DEBUG')
    ),
    FrameworkJobReport = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    suppressWarning = cms.untracked.vstring(),
    statistics = cms.untracked.vstring('cerr_stats'),
    cerr_stats = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        output = cms.untracked.string('cerr')
    ),
    infos = cms.untracked.PSet(
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        placeholder = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('warnings', 
        'errors', 
        'infos', 
        'debugs', 
        'cout', 
        'cerr'),
    debugModules = cms.untracked.vstring('bscTrigger'),
    categories = cms.untracked.vstring('BscSim','FwkJob', 
        'FwkReport', 
        'FwkSummary', 
        'Root_NoDictionary'),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport')
)
# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)




# Event output

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('BSCTrigger.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('trigger_step')
    )
)

# Path and EndPath definitions

process.trigger_step = cms.Path(process.bscTrigger)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.trigger_step,process.outpath)




