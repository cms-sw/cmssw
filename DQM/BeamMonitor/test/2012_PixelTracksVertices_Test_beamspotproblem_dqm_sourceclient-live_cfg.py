import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
###DQM Envirionment####################
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQM.BeamMonitor.BeamSpotProblemMonitor_cff")

####---Massage logger Services----####
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['BeamSpotProblemMonitor']
process.MessageLogger.categories = ['BeamSpotProblemMonitor']
process.MessageLogger.cerr.threshold = "INFO"


#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamSpotProblemMonitor'

import DQMServices.Components.DQMEnvironment_cfi
process.load("DQM.Integration.test.inputsource_cfi")
process.DQMEventStreamHttpReader.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('HLT*',
                                        'HLT_Jet*',
                                        'HLT_*Cosmic*',
                                        'HLT_HT*',
                                        'HLT_MinBias_*',
                                        'HLT_Physics*',
                                        'HLT_ZeroBias*'))

#----Event to Analyze---------------
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
  #RAW, Run2012A, MinBias Run 190782
  'file:/tmp/schauhan/Run190782MinmumBias/CAB7DEAE-B782-E111-8395-0030486733B4.root',
  'file:/tmp/schauhan/Run190782MinmumBias/8E834D27-B982-E111-A7B0-00215AEDFD98.root'
  #MCTest:QCD relval 8 TeV
  #'file:/tmp/schauhan/MCRelval52X/348A1DE4-0A7A-E111-892A-003048FFCC0A.root',
  #'file:/tmp/schauhan/MCRelval52X/4C132D2F-0F7A-E111-86C6-0018F3D09628.root',
  #'file:/tmp/schauhan/MCRelval52X/A019F59E-0E7A-E111-A37C-0018F3D09614.root'
  ##MCTest:QCD relval 7 TeV
  #'file:/tmp/schauhan/MCRelval44X/22C9A2AA-17E6-E011-BCE5-0018F3D09620.root',
  #'file:/tmp/schauhan/MCRelval44X/7E44063F-1BE6-E011-B2EA-001A928116CC.root',
  #'file:/tmp/schauhan/MCRelval44X/C09174A3-0EE6-E011-9145-003048678F74.root'


 ),
    skipBadFiles = cms.untracked.bool(True),  
)


#--------------------------
# Filters
#--------------------------
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
process.hltTriggerTypeFilter.SelectedTriggerType = 1

#----------------------------
# BeamMonitor
#-----------------------------
process.dqmBeamSpotProblemMonitor.OnlineMode        = True
process.dqmBeamSpotProblemMonitor.AlarmONThreshold  = 10
process.dqmBeamSpotProblemMonitor.AlarmOFFThreshold = 12

#number of minimum cosmic trks to avoid false alarm
process.dqmBeamSpotProblemMonitor.nCosmicTrk        = 10

process.dqmBeamSpotProblemMonitor.doTest        = True

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-----------------------------
# Magnetic Field
#-----------------------------
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
## Collision Reconstruction
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi")
# Offline Beam Spot
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")






#----------------------------           
# Pixel-Tracks-Vertices Configuration            
#----------------------------           

process.pixelVertices.TkFilterParameters.minPt = process.pixelTracks.RegionFactoryPSet.RegionPSet.ptMin

process.offlinePrimaryVertices.TrackLabel = cms.InputTag("pixelTracks")

process.tracking_FirstStep  = cms.Sequence(    process.siPixelDigis* 
                                               process.offlineBeamSpot*
                                               process.siPixelClusters*
                                               process.siPixelRecHits*
                                               process.pixelTracks*
                                               process.pixelVertices
                                           )

#--pixel tracking ends here-----

from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
                                 qtList = cms.untracked.FileInPath('DQM/BeamMonitor/test/BeamSpotAvailableTest.xml'),
                                 prescaleFactor = cms.untracked.int32(1),                               
                                 testInEventloop = cms.untracked.bool(False),
                                 verboseQT =  cms.untracked.bool(True)                 
                                )

process.dqmcommon = cms.Sequence( process.qTester
                                 *process.dqmEnv
                                 *process.dqmSaver)

process.monitor = cms.Sequence(process.dqmBeamSpotProblemMonitor
                               )


# Setup DQM store parameters.
process.DQMStore.verbose = 1
process.DQM.collectorHost   = 'lxplus414.cern.ch'
process.DQM.collectorPort   = 9190
process.dqmSaver.dirName    = '.'
process.dqmSaver.producer   = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamSpotProblemMonitor'
process.dqmSaver.saveByRun     = 1
process.dqmSaver.saveAtJobEnd  = True

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_P_V32::All'


process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# reduce verbosity
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)


process.pp = cms.Path( 
                      process.scalersRawToDigi
                      *process.hltTriggerTypeFilter
                      *process.dqmcommon
                      *process.tracking_FirstStep
                      *process.monitor
                     )


process.castorDigis.InputLabel = cms.InputTag("rawDataCollector")
process.csctfDigis.producer = cms.InputTag("rawDataCollector")
process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataCollector")
process.ecalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataCollector")
process.gctDigis.inputLabel = cms.InputTag("rawDataCollector")
process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataCollector")
process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
process.muonDTDigis.inputLabel = cms.InputTag("rawDataCollector")
process.muonRPCDigis.InputLabel = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")
