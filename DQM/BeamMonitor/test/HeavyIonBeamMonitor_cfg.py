import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.BeamMonitor.BeamMonitor_cff")
process.load("DQM.BeamMonitor.BeamMonitorBx_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")

process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff") ## For Data
#process.load("Configuration.StandardSequences.RawToDigi_cff")     ## For MC
process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff") ## HI sequences
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['dqmBeamMonitor']
process.MessageLogger.BeamMonitor = dict()
process.MessageLogger.cerr.threshold = "INFO"

#--------------------------
# Input files (heavy-ion MC)
#--------------------------

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        # Virgin raw
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/FAD966AE-64C6-DF11-9649-0019B9F72BFF.root',
        '/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/F6C87AAF-69C6-DF11-A9D0-001D09F23944.root',
    )
)

#--------------------------
# Filters
#--------------------------

# HLT Filter
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1

# L1 Trigger Bit Selection (bit 40 and 41 for BSC trigger)
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('40 OR 41')

# Select events based on the pixel cluster multiplicity
import  HLTrigger.special.hltPixelActivityFilter_cfi
process.multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
    inputTag  = cms.InputTag('siPixelClusters'),
    minClusters = cms.uint32(150),
    maxClusters = cms.uint32(50000)
    )

process.phystrigger = cms.Sequence(
                                   process.hltTriggerTypeFilter*
                                   process.gtDigis*
                                   process.hltLevel1GTSeed)

process.filter_step = cms.Sequence( process.siPixelDigis
                                   *process.siPixelClusters
                                   #*process.multFilter
                                  )


process.RecoForDQM = cms.Sequence(*process.siPixelDigis
                                  *process.siPixelClusters
                                  *process.siPixelRecHits
                                  *process.offlineBeamSpot
                                  *process.hiPixelVertices
                                  *process.hiPixel3PrimTracksSequence
                                 )


# make pixel vertexing less sensitive to incorrect beamspot
process.hiPixel3ProtoTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.2
process.hiPixel3ProtoTracks.RegionFactoryPSet.RegionPSet.fixedError = 0.5
process.hiSelectedProtoTracks.maxD0Significance = 100
process.hiPixelAdaptiveVertex.TkFilterParameters.maxD0Significance = 100
process.hiPixelAdaptiveVertex.useBeamConstraint = False
process.hiPixelAdaptiveVertex.PVSelParameters.maxDistanceToBeam = 1.0


# use HI pixel tracking and vertexing
process.dqmBeamMonitor.BeamFitter.TrackCollection = cms.untracked.InputTag('hiPixel3PrimTracks')
process.dqmBeamMonitor.primaryVertex = cms.untracked.InputTag('hiSelectedVertex')
process.dqmBeamMonitor.PVFitter.VertexCollection = cms.untracked.InputTag('hiSelectedVertex')


# Change Beam Monitor variables
if process.dqmSaver.producer.value() is "Playback":
  process.dqmBeamMonitor.BeamFitter.WriteAscii = False
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = 'BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = 'DIP_BeamFitResults.txt'
else:
  process.dqmBeamMonitor.BeamFitter.WriteAscii = True
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = 'BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = 'DIP_BeamFitResults.txt'
#process.dqmBeamMonitor.BeamFitter.SaveFitResults = False
#process.dqmBeamMonitor.BeamFitter.OutputFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.root'
  process.dqmBeamMonitorBx.BeamFitter.WriteAscii = True
  process.dqmBeamMonitorBx.BeamFitter.AsciiFileName = 'BeamFitResults_Bx.txt'
 
# Lower for HI
process.dqmBeamMonitor.PVFitter.minNrVerticesForFit   = 20
process.dqmBeamMonitorBx.PVFitter.minNrVerticesForFit = 20
 
 
## TKStatus
process.dqmTKStatus = cms.EDAnalyzer("TKStatus",
        BeamFitter = cms.PSet(
        DIPFileName = process.dqmBeamMonitor.BeamFitter.DIPFileName
        )
)
 
process.dqmBeamMonitor.OnlineMode = True                  ## in MC the LS are not ordered??
process.dqmBeamMonitor.BeamFitter.MinimumTotalLayers = 3   ## using pixel triplets
process.dqmBeamMonitor.resetEveryNLumi = 10                ## default is 20
process.dqmBeamMonitor.resetPVEveryNLumi = 10               ## default is 5

process.load("FWCore.Modules.preScaler_cfi")
process.preScaler.prescaleFactor = 300

#--------------------------
# DQM output
#--------------------------

# Setup DQM store parameters.
process.DQMStore.verbose = 0
process.DQM.collectorHost = 'lxplus438.cern.ch'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_44_V4::All' ## 'realistic' offline beamspot, unlike 36x sample

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )


process.hi = cms.Path(*process.preScaler
                      *process.hltTriggerTypeFilter
                      *process.RawToDigi
                      *process.filter_step
                      *process.RecoForDQM
                      *process.dqmBeamMonitor
                      +process.dqmEnv
                      +process.dqmSaver)


process.schedule = cms.Schedule(process.hi)

