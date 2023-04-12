import FWCore.ParameterSet.Config as cms

process = cms.Process("DTMonitorStream")

# the source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Run2022B/HLTPhysics/RAW/v1/000/355/456/00000/69b26b27-4bd1-4524-bc18-45f7b9b5e076.root',
    ),
    skipEvents = cms.untracked.uint32(0)
)

process.maxEvents.input = 100

process.load("HLTrigger.special.hltDTROMonitorFilter_cfi")
process.hltDTROMonitorFilter.inputLabel = 'rawDataCollector'

# # message logger
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('DTDataIntegrityTask'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(False),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        DTDataIntegrityTask = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

# --- Timing --------------------------------------------------------------
# process.load("HLTrigger.Timer.timer_cfi")
# process.TimerService = cms.Service("TimerService",
#     useCPUtime = cms.untracked.bool(True)
# )
# process.pts = cms.EDFilter("PathTimerInserter")
# process.PathTimerService = cms.Service("PathTimerService")
# -------------------------------------------------------------------------

# --- Output Module -------------------------------------------------------
process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_fedSelector_*_*'
    ),
    fileName = cms.untracked.string('dtDebugStream.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('dtROMonitorSelection')
    )
)
# -------------------------------------------------------------------------

process.dtROMonitorSelection = cms.Path(process.hltDTROMonitorFilter)

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE'),
    wantSummary = cms.untracked.bool(True)
)

process.outpath = cms.EndPath(process.out)
