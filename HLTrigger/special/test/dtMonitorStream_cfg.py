import FWCore.ParameterSet.Config as cms

process = cms.Process("DTMonitorStream")

# the source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/F04B7D65-4E67-DE11-AFE3-000423D952C0.root',
    '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/ECC3F663-4E67-DE11-ABD0-000423D94A20.root',
    '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/D650D167-4E67-DE11-853D-000423D98A44.root',
    '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/D29CE383-4B67-DE11-9FF6-000423D951D4.root',
    '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/CCC5CAD4-4A67-DE11-B9CB-000423D99660.root',
    '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/8C905A66-4E67-DE11-9A59-000423D6A6F4.root',
    '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/20221969-4E67-DE11-915E-000423D99BF2.root'
   ),
                            skipEvents = cms.untracked.uint32(95000)                       
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
    )


process.load("Configuration.StandardSequences.Geometry_cff")


# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
process.GlobalTag.globaltag = "GR09_31X_V2P::All"
#process.prefer("GlobalTag")

process.load("HLTrigger.special.hltDTROMonitorFilter_cfi")
# process.dtmonitorfilterCached = cms.EDFilter("HLTDTROMonitorFilter")

process.fedSelector = cms.EDProducer("SubdetFEDSelector",
                                     rawInputLabel = cms.InputTag("source"),
                                     getECAL = cms.bool(False),
                                     getSiStrip = cms.bool(False),
                                     getSiPixel = cms.bool(False),
                                     getHCAL = cms.bool(False),
                                     getMuon = cms.bool(True),
                                     getTrigger = cms.bool(False)
                                     )



# # message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring('*'),
                                    destinations = cms.untracked.vstring('cout'),
                                    categories = cms.untracked.vstring('DTDataIntegrityTask'), 
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'),
                                                              noLineBreaks = cms.untracked.bool(False),
                                                              DEBUG = cms.untracked.PSet(
    limit = cms.untracked.int32(0)),
                                                              INFO = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)),
                                                              DTDataIntegrityTask = cms.untracked.PSet(
    limit = cms.untracked.int32(-1))
                                                              )
                                    )






# --- Timing --------------------------------------------------------------
# process.load("HLTrigger.Timer.timer_cfi")
# process.TimerService = cms.Service("TimerService",
#                                    useCPUtime = cms.untracked.bool(True)
#                                    )
# process.pts = cms.EDFilter("PathTimerInserter")
# process.PathTimerService = cms.Service("PathTimerService")
# -------------------------------------------------------------------------


# --- Output Module -------------------------------------------------------
process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *', 
                                                                      'keep *_fedSelector_*_*'),
                               fileName = cms.untracked.string('dtDebugStream.root'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('dtROMonitorSelection'))
                               )

# -------------------------------------------------------------------------

# process.dtROMonitorSelection = cms.Path(process.dtmonitorfilter + process.dtmonitorfilterCached + process.pts + process.fedSelector)

process.dtROMonitorSelection = cms.Path(process.dtmonitorfilter + process.fedSelector)


process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE'),
    wantSummary = cms.untracked.bool(True)
    )

process.outpath = cms.EndPath(process.out)

