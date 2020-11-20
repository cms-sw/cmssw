import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
###DQM Envirionment####################
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.BeamMonitor.BeamMonitor_Pixel_cff")
process.load("DQM.BeamMonitor.BeamMonitorBx_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")


####---Massage logger Services----####
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['dqmBeamMonitor']
process.MessageLogger.BeamMonitor = dict()
process.MessageLogger.cerr.threshold = "INFO"


#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamMonitor'

import DQMServices.Components.DQMEnvironment_cfi
process.dqmEnvPixelLess = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
process.dqmEnvPixelLess.subSystemFolder = 'BeamMonitor_PixelLess'


process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('HLT_L1*',
                               'HLT_ZeroBias*',
                               'HLT_Jet*'))


#----Event to Analyze---------------
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
   #HighPU Fill 2011:Run 178203
   #'file:/tmp/schauhan/HighPUHPF/BC4C924F-4EF3-E011-B001-E0CB4E553651.root',
   #'file:/tmp/schauhan/HighPUHPF/B68B5DDB-4FF3-E011-AD2B-001D09F2B30B.root',
   #'file:/tmp/schauhan/HighPUHPF/284D89DB-4FF3-E011-9B4E-001D09F28EA3.root'   
   #'file:/tmp/schauhan/RAWRECO/88B58FFA-0B21-E111-958E-002618943905.root'
   #Run 177515, Normal 2011B run RAW file
   #'file:/tmp/schauhan/Run2011B_MinimumBias_RAW/1431D41F-E5EA-E011-8F71-001D09F291D2.root'
   #Run 177139, Normal 2011B run RAW file                              
   'file:/tmp/schauhan/Run2011B_MinimumBias_RAW/46A6F982-BDE7-E011-8172-BCAEC53296FF_177139_LS400_420.root'
   #Raw, 178208 highpileup
   #'file:/tmp/schauhan/HighPileUp_Run178208/A2CBA95C-60F3-E011-B4A5-001D09F251CC.root',
   #'file:/tmp/schauhan/HighPileUp_Run178208/BC9AC280-62F3-E011-B751-BCAEC518FF52.root',
   #'file:/tmp/schauhan/HighPileUp_Run178208/50455D75-62F3-E011-BF70-0015C5FDE067.root',
   #'file:/tmp/schauhan/HighPileUp_Run178208/30F48998-60F3-E011-80C0-003048CF99BA.root'
   #'file:/tmp/schauhan/HighPileUp_Run178208/EC9C9C74-62F3-E011-B0EE-0019B9F4A1D7.root'

 ),
    skipBadFiles = cms.untracked.bool(True),  
)


#--------------------------
# Filters
#--------------------------
# HLT Filter
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1

#----------------------------
# BeamMonitor
#-----------------------------
process.dqmBeamMonitor.OnlineMode        = True

process.dqmBeamMonitor.resetEveryNLumi   = 5
process.dqmBeamMonitor.resetPVEveryNLumi = 5
process.dqmBeamMonitor.PVFitter.errorScale = 1.23

process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 20




#TriggerName for selecting pv for DIP
process.dqmBeamMonitor.jetTrigger  = cms.untracked.vstring("HLT_L1",
                                                           "HLT_Act")

process.dqmBeamMonitor.hltResults = cms.InputTag("TriggerResults","","HLT")
#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-----------------------------
# Magnetic Field
#-----------------------------
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')



#-----------------------
#  Reconstruction Modules
#-----------------------
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
  process.dqmBeamMonitorBx.BeamFitter.WriteAscii = True
  process.dqmBeamMonitorBx.BeamFitter.AsciiFileName = 'BeamFitResults_Bx.txt'

# 


## TKStatus
process.dqmTKStatus = cms.EDAnalyzer("TKStatus",
        BeamFitter = cms.PSet(
        DIPFileName = process.dqmBeamMonitor.BeamFitter.DIPFileName
        )
)



process.dqmcommon = cms.Sequence(process.dqmEnv
                                 *process.dqmSaver)

process.monitor = cms.Sequence(process.dqmBeamMonitor
                               )



# Setup DQM store parameters.
process.DQMStore.verbose = 1
process.DQM.collectorHost   = 'lxplus414.cern.ch'
process.DQM.collectorPort   = 9190
process.dqmSaver.dirName    = '.'
process.dqmSaver.producer   = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun     = 1
process.dqmSaver.saveAtJobEnd  = True

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_52_V2::All'


process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# reduce verbosity
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1)


process.pp = cms.Path( process.scalersRawToDigi
                      *process.dqmTKStatus
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


 
#print process.dumpPython()
