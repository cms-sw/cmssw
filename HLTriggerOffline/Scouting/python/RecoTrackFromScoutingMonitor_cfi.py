import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackingMonitor_cfi import TrackMon as _TrackMon
scoutingRecoTrackMonitor = _TrackMon.clone(
    allTrackProducer = cms.InputTag("recoTracksFromScouting"),
    TrackProducer    = cms.InputTag("recoTracksFromScouting"),
    SeedProducer     = cms.InputTag(""),
    TCProducer       = cms.InputTag(""),
    MVAProducers     = cms.vstring(""),
    TrackProducerForMVA = cms.InputTag(""),
    ClusterLabels    = cms.vstring(''),
    beamSpot         = cms.InputTag("hltOnlineBeamSpot"),
    primaryVertex    = cms.InputTag("recoVerticesFromScouting"),
    stripCluster     = cms.InputTag(''),
    pixelCluster     = cms.InputTag(''),                          
    BXlumiSetup      = cms.PSet(),
    genericTriggerEventPSet = cms.PSet(),
    
    # PU monitoring
    primaryVertexInputTags    = cms.VInputTag(),
    selPrimaryVertexInputTags = cms.VInputTag(),
    pvLabels = cms.vstring(),
                          
    # output parameters
    AlgoName            = cms.string('ScoutingTk'),
    Quality             = cms.string(''),
    FolderName          = cms.string('HLT/ScoutingOffline/Tracks'),
    BSFolderName        = cms.string('HLT/ScoutingOffline/Tracks'),
    PVFolderName        = cms.string('HLT/ScoutingOffline/Tracks'),

    # determines where to evaluate track parameters
    # options: 'default'      --> straight up track parametes
    #		   'ImpactPoint'  --> evalutate at impact point 
    #		   'InnerSurface' --> evalutate at innermost measurement state 
    #		   'OuterSurface' --> evalutate at outermost measurement state 

    MeasurementState = cms.string('ImpactPoint'),
    
    # which plots to do

    doAllPlots                          = cms.bool(False),
    doGeneralPropertiesPlots            = cms.bool(True),   
    doHitPropertiesPlots                = cms.bool(True),
    doRecHitVsPhiVsEtaPerTrack          = cms.bool(True),
    doRecHitVsPtVsEtaPerTrack           = cms.bool(True), 
    doLayersVsPhiVsEtaPerTrack          = cms.bool(True),  
    doRecHitsPerTrackProfile            = cms.bool(True),     
    doBeamSpotPlots                     = cms.bool(True),  
    doPrimaryVertexPlots                = cms.bool(True),   
    doDCAPlots                          = cms.bool(True),  
    doDCAwrtPVPlots                     = cms.bool(True),   
    doDCAwrt000Plots                    = cms.bool(True),
    doSIPPlots                          = cms.bool(True),
    doThetaPlots                        = cms.bool(True),
    doTrackPxPyPlots                    = cms.bool(True),
    doLumiAnalysis                      = cms.bool(True),
    doProfilesVsLS                      = cms.bool(True),
    doPlotsVsGoodPVtx                   = cms.bool(False),
    doEffFromHitPatternVsPU             = cms.bool(False),
    doEffFromHitPatternVsBX             = cms.bool(False),
    doEffFromHitPatternVsLUMI           = cms.bool(False),
    doSeedParameterHistos               = cms.bool(False),
    doTrackCandHistos                   = cms.bool(False),
    doAllTrackCandHistos                = cms.bool(False),
    doTestPlots                         = cms.bool(False),
    doTrackerSpecific                   = cms.bool(False),
    doMeasurementStatePlots             = cms.bool(False),
    pvNDOF                              = cms.int32(4),
    pixelCluster4lumi                   = cms.InputTag(''),
    scal                                = cms.InputTag(''),
    forceSCAL                           = cms.bool(False),
    metadata                            = cms.InputTag('hltOnlineMetaDataDigis'),
)
