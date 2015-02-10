import FWCore.ParameterSet.Config as cms

process = cms.Process("BeamMonitor")

#-------------------------------------
# ALL Common Stuff for PP and H.I
#-------------------------------------
process.load("DQM.Integration.test.inputsource_cfi")

#--------------------------
# Filters
#--------------------------
# HLT Filter
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1


#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamMonitor'

import DQMServices.Components.DQMEnvironment_cfi
process.dqmEnvPixelLess = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
process.dqmEnvPixelLess.subSystemFolder = 'BeamMonitor_PixelLess'

#----------------------------
# BeamMonitor
#-----------------------------
process.load("DQM.BeamMonitor.BeamMonitor_cff")
process.load("DQM.BeamMonitor.BeamMonitorBx_cff")
process.load("DQM.BeamMonitor.BeamMonitor_PixelLess_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")

#---------------------------
# Reco reltaed
#---------------------------

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

# Change Beam Monitor variables
if process.dqmSaver.producer.value() is "Playback":
  process.dqmBeamMonitor.BeamFitter.WriteAscii = False
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmdev/BeamMonitorDQM/BeamFitResults.txt'
else:
  process.dqmBeamMonitor.BeamFitter.WriteAscii = True
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults.txt'
#process.dqmBeamMonitor.BeamFitter.SaveFitResults = False
#process.dqmBeamMonitor.BeamFitter.OutputFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.root'
  process.dqmBeamMonitorBx.BeamFitter.WriteAscii = True
  process.dqmBeamMonitorBx.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults_Bx.txt'


## TKStatus
process.dqmTKStatus = cms.EDAnalyzer("TKStatus",
        BeamFitter = cms.PSet(
        DIPFileName = process.dqmBeamMonitor.BeamFitter.DIPFileName
        )
)



process.dqmcommon = cms.Sequence(process.dqmEnv
                                 *process.dqmSaver)

process.monitor = cms.Sequence(process.dqmBeamMonitor)



#--------------------------
# Proton Proton Stuff
#--------------------------

if (process.runType.getRunType() == process.runType.pp_run or process.runType.getRunType() == process.runType.cosmic_run):
    print "Running pp paths"

    process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(
              SelectEvents = cms.vstring('HLT_L1*',
                                         'HLT_Jet*',
                                         'HLT_*Cosmic*',
                                         'HLT_HT*',
                                         'HLT_MinBias_*',
                                         'HLT_Physics*',
                                         'HLT_ZeroBias_v*'))

    process.load("Configuration.StandardSequences.Reconstruction_cff")

    #Offline Beam Spot
    process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

    ## Offline PrimaryVertices
    import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
    process.offlinePrimaryVertices = RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()
    process.offlinePrimaryVertices.TrackLabel = cms.InputTag("firstStepTracksWithQuality")
    process.dqmBeamMonitor.BeamFitter.TrackCollection = cms.untracked.InputTag('firstStepTracksWithQuality')
    process.dqmBeamMonitorBx.BeamFitter.TrackCollection = cms.untracked.InputTag('firstStepTracksWithQuality')
    #### END OF TRACKING RECONSTRUCTION ####

    process.dqmBeamMonitor.OnlineMode = True                  ## in MC the LS are not ordered??
    process.dqmBeamMonitor.resetEveryNLumi = 5
    process.dqmBeamMonitor.resetPVEveryNLumi = 5
    process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 25

    #TriggerName for selecting pv for DIP publication, NO wildcard needed here
    #it will pick all triggers which has these strings in theri name
    process.dqmBeamMonitor.jetTrigger  = cms.untracked.vstring("HLT_ZeroBias_v",
                                                               "HLT_Jet300_v",
                                                               "HLT_QuadJet70_v")

    process.dqmBeamMonitor.hltResults = cms.InputTag("TriggerResults","","HLT")

    process.tracking_FirstStep = cms.Sequence(process.siPixelDigis
                                         *process.siStripDigis
                                         *process.trackerlocalreco
                                         *process.offlineBeamSpot
                                         *process.recopixelvertexing
                                         *process.firstStep)


    process.p = cms.Path(process.scalersRawToDigi
                         *process.dqmTKStatus
                         *process.hltTriggerTypeFilter
                         *process.dqmcommon
                         *process.tracking_FirstStep
                         *process.offlinePrimaryVertices
                         *process.monitor)


#--------------------------------------------------
# Heavy Ion Stuff
#--------------------------------------------------

if (process.runType.getRunType() == process.runType.hi_run):
   
    print "Running HI paths"
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")


#----------------------------                                                                                                                                                      
# Event Source
#-----------------------------
    process.EventStreamHttpReader.SelectEvents =  cms.untracked.PSet(
              SelectEvents = cms.vstring('HLT_HI*')
            )
 
    # Beamspot DQM options
    process.dqmBeamMonitor.OnlineMode = True                  
    process.dqmBeamMonitor.resetEveryNLumi = 10
    process.dqmBeamMonitor.resetPVEveryNLumi = 10
    process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 20
    process.dqmBeamMonitor.BeamFitter.MinimumTotalLayers = 3   ## using pixel triplets

    #TriggerName for selecting pv for DIP publication, NO wildcard needed here
    #it will pick all triggers which has these strings in theri name
    process.dqmBeamMonitor.jetTrigger  = cms.untracked.vstring("HLT_HI")
                                                               
    process.dqmBeamMonitor.hltResults = cms.InputTag("TriggerResults","","HLT")
 
    ## Load Heavy Ion Sequence
    process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff") 
 
    # Select events based on the pixel cluster multiplicity
    import  HLTrigger.special.hltPixelActivityFilter_cfi
    process.multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
                         inputTag  = cms.InputTag('siPixelClusters'),
                         minClusters = cms.uint32(150),
                         maxClusters = cms.uint32(50000)
                )
 
    process.filter_step = cms.Sequence( process.siPixelDigis
                                       *process.siPixelClusters
                                       #*process.multFilter
                                  )

    process.HIRecoForDQM = cms.Sequence( process.siPixelDigis
                                        *process.siPixelClusters
                                        *process.siPixelRecHits
                                        *process.offlineBeamSpot
                                        *process.hiPixelVertices
                                        *process.hiPixel3PrimTracks
                                   )
 
    # use HI pixel tracking and vertexing
    process.dqmBeamMonitor.BeamFitter.TrackCollection = cms.untracked.InputTag('hiPixel3PrimTracks')
    process.dqmBeamMonitorBx.BeamFitter.TrackCollection = cms.untracked.InputTag('hiPixel3PrimTracks')
    process.dqmBeamMonitor.primaryVertex = cms.untracked.InputTag('hiSelectedVertex')
    process.dqmBeamMonitor.PVFitter.VertexCollection = cms.untracked.InputTag('hiSelectedVertex')
 
 
    # make pixel vertexing less sensitive to incorrect beamspot
    process.hiPixel3ProtoTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.2
    process.hiPixel3ProtoTracks.RegionFactoryPSet.RegionPSet.fixedError = 0.5
    process.hiSelectedProtoTracks.maxD0Significance = 100
    process.hiPixelAdaptiveVertex.TkFilterParameters.maxD0Significance = 100
    process.hiPixelAdaptiveVertex.useBeamConstraint = False
    process.hiPixelAdaptiveVertex.PVSelParameters.maxDistanceToBeam = 1.0
 
 
    #--------------------------
    # Scheduling             
    #--------------------------
                         
    process.hi = cms.Path(process.scalersRawToDigi
                        *process.dqmTKStatus
                        *process.hltTriggerTypeFilter
                        *process.filter_step
                        *process.HIRecoForDQM
                        *process.dqmcommon
                        *process.monitor)         
