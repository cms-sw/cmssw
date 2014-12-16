# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step1 -s DQM -n 1 --eventcontent DQM --conditions auto:com10 --filein /store/relval/CMSSW_7_1_2/MinimumBias/RECO/GR_R_71_V7_dvmc_RelVal_mb2012Cdvmc-v1/00000/00209DF4-3708-E411-9FA7-0025905A6126.root --data --no_exec --python_filename=test_step1_cfg.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('DQM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
  secondaryFileNames = cms.untracked.vstring(),
  fileNames = cms.untracked.vstring([
#    '/store/relval/CMSSW_7_1_1/RelValZEEdvmc/GEN-SIM-RECO/PU50ns_START71_V8A_rundepMC203002_dvmc-v2/00000/006BE001-5900-E411-967F-0025905A60D0.root'
     '/store/relval/CMSSW_7_3_0_pre2/RelValZEE_13/GEN-SIM-RECO/PU50ns_MCRUN2_73_V0-v1/00000/0A397986-F26B-E411-A3F8-02163E00EA5D.root'
  ])
)

process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step1 nevts:1'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('step1_DQM_1.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'START71_V8A::All', '')

# Tracker Data MC validation suite
process.load('DQM.TrackingMonitorSource.TrackingDataMCValidation_Standalone_cff')

## Primary Vertex Selector
#process.selectedPrimaryVertices = cms.EDFilter("VertexSelector",
#   src = cms.InputTag('offlinePrimaryVertices'),
#   cut = cms.string("!isFake && ndof >= 4 && abs(z) < 24 && abs(position.Rho) < 2.0"),
#   filter = cms.bool(True)
#)
#process.selectedTracks = cms.EDFilter("TrackSelector",
#   src = cms.InputTag('generalTracks'),	
#   cut = cms.string("pt > 1.0"),
#   filter = cms.bool(True)    
#)
#process.trackTypeMonitor = cms.EDAnalyzer('TrackTypeMonitor',
#    trackInputTag   = cms.untracked.InputTag('selectedTracks'),
#    offlineBeamSpot = cms.untracked.InputTag('offlineBeamSpot'),
#    trackQuality    = cms.untracked.string('highPurity'),
#    vertexTag       = cms.untracked.InputTag('selectedPrimaryVertices'),
#    # isMC            = cms.untracked.bool(True),
#    # PUCorrection    = cms.untracked.bool(False),
#    TrackEtaPar    = cms.PSet(Xbins = cms.int32(60),Xmin = cms.double(-3.0),Xmax = cms.double(3.0)),
#    TrackPtPar     = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(0.0),Xmax = cms.double(100.0)),
#    TrackPPar      = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(0.0),Xmax = cms.double(100.0)),
#    TrackPhiPar    = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(-4.0),Xmax = cms.double(4.0)),
#    TrackPterrPar  = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(0.0),Xmax = cms.double(100.0)),
#    TrackqOverpPar = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(-10.0),Xmax = cms.double(10.0)),
#    TrackdzPar     = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(-100.0),Xmax = cms.double(100.0)),
#    TrackChi2bynDOFPar = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(0.0),Xmax = cms.double(10.0)),
#    nTracksPar     = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(-0.5),Xmax = cms.double(99.5))
#)
#process.hltEventAnalyzer = cms.EDAnalyzer("HLTEventAnalyzer",
#    processName = cms.string("HLT"),
#    triggerName = cms.string("@"),
#    triggerResults = cms.InputTag("TriggerResults","","HLT"),
#    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")
#)
#process.hltEventFilter = cms.EDFilter("HLTEventFilter",
#    processName = cms.string("HLT"),
#    triggerName = cms.string("HLT_ZeroBias_v7"),
#    triggerResults = cms.InputTag("TriggerResults","","HLT"),
#    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")
#)
#process.Tracer = cms.Service("Tracer")
#process.ztoMMEventSelector = cms.EDFilter("ZtoMMEventSelector")
#process.ztoEEEventSelector = cms.EDFilter("ZtoEEEventSelector")

#process.analysis_step = cms.Path(process.ztoEEEventSelector
#                               * process.selectedPrimaryVertices
#                               * process.selectedTracks
#                               * process.trackTypeMonitor)

process.analysis_step = cms.Path(  process.selectedTracks
                                 * process.selectedPrimaryVertices
                                 * process.ztoEEEventSelector 
                                 * process.standaloneTrackMonitorMM)
#process.analysis_step = cms.Path(process.standaloneTrackMonitorMM)
# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step, process.endjob_step, process.DQMoutput_step)
