import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackingMonitor_cfi
TrackerCollisionTrackMon = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()

# Update specific parameters

# input tags
TrackerCollisionTrackMon.TrackProducer          = cms.InputTag("generalTracks")
TrackerCollisionTrackMon.SeedProducer           = cms.InputTag("initialStepSeeds")
TrackerCollisionTrackMon.TCProducer             = cms.InputTag("initialStepTrackCandidates")
TrackerCollisionTrackMon.ClusterLabels          = cms.vstring('Tot','Strip','Pix') # to decide which Seeds-Clusters correlation plots to have default is Total other options 'Strip', 'Pix'
TrackerCollisionTrackMon.beamSpot               = cms.InputTag("offlineBeamSpot")
TrackerCollisionTrackMon.primaryVertex          = cms.InputTag('offlinePrimaryVertices')
TrackerCollisionTrackMon.primaryVertexInputTags = cms.VInputTag(
      cms.InputTag('offlinePrimaryVertices')
)    
TrackerCollisionTrackMon.selPrimaryVertexInputTags = cms.VInputTag(
      cms.InputTag('goodOfflinePrimaryVertices')
)    
TrackerCollisionTrackMon.pvLabels               = cms.vstring(
      'offline'
)

# output parameters
TrackerCollisionTrackMon.AlgoName              = cms.string('GenTk')
TrackerCollisionTrackMon.Quality               = cms.string('')
TrackerCollisionTrackMon.FolderName            = cms.string('Tracking/GlobalParameters')
TrackerCollisionTrackMon.BSFolderName          = cms.string('Tracking/ParametersVsBeamSpot')

# determines where to evaluate track parameters
# 'ImpactPoint'  --> evalutate at impact point 
TrackerCollisionTrackMon.MeasurementState      = cms.string('ImpactPoint')

# which plots to do
TrackerCollisionTrackMon.doAllPlots                          = cms.bool(False)
TrackerCollisionTrackMon.doGoodTrackPlots                    = cms.bool(True)
TrackerCollisionTrackMon.doTrackerSpecific                   = cms.bool(True)
TrackerCollisionTrackMon.doHitPropertiesPlots                = cms.bool(True)
TrackerCollisionTrackMon.doGeneralPropertiesPlots            = cms.bool(True)
TrackerCollisionTrackMon.doBeamSpotPlots                     = cms.bool(True)
TrackerCollisionTrackMon.doSeedParameterHistos               = cms.bool(False)
TrackerCollisionTrackMon.doRecHitVsPhiVsEtaPerTrack          = cms.bool(True)
TrackerCollisionTrackMon.doGoodTrackRecHitVsPhiVsEtaPerTrack = cms.bool(True)
TrackerCollisionTrackMon.doLayersVsPhiVsEtaPerTrack          = cms.bool(True)
TrackerCollisionTrackMon.doGoodTrackLayersVsPhiVsEtaPerTrack = cms.bool(True)
TrackerCollisionTrackMon.doPUmonitoring                      = cms.bool(False)
TrackerCollisionTrackMon.doPlotsVsBXlumi                     = cms.bool(False)
TrackerCollisionTrackMon.doPlotsVsGoodPVtx                   = cms.bool(True)
TrackerCollisionTrackMon.doEffFromHitPattern                 = cms.bool(True)

# LS analysis
TrackerCollisionTrackMon.doLumiAnalysis       = cms.bool(True)     
TrackerCollisionTrackMon.doProfilesVsLS       = cms.bool(True)

TrackerCollisionTrackMon.doSeedNumberHisto    = cms.bool(False)
TrackerCollisionTrackMon.doSeedETAHisto       = cms.bool(False)
TrackerCollisionTrackMon.doSeedVsClusterHisto = cms.bool(False)

# Number of Tracks per Event
TrackerCollisionTrackMon.TkSizeBin             = cms.int32(200)
TrackerCollisionTrackMon.TkSizeMax             = cms.double(999.5)                        
TrackerCollisionTrackMon.TkSizeMin             = cms.double(-0.5)

# chi2 dof
TrackerCollisionTrackMon.Chi2NDFBin            = cms.int32(50)
TrackerCollisionTrackMon.Chi2NDFMax            = cms.double(49.5)
TrackerCollisionTrackMon.Chi2NDFMin            = cms.double(-0.5)

# Number of seeds per Event
TrackerCollisionTrackMon.TkSeedSizeBin = cms.int32(100)
TrackerCollisionTrackMon.TkSeedSizeMax = cms.double(499.5)                        
TrackerCollisionTrackMon.TkSeedSizeMin = cms.double(-0.5)

# Number of Track Cadidates per Event
TrackerCollisionTrackMon.TCSizeBin = cms.int32(100)
TrackerCollisionTrackMon.TCSizeMax = cms.double(499.5)
TrackerCollisionTrackMon.TCSizeMin = cms.double(-0.5)

TrackerCollisionTrackMon.GoodPVtxBin = cms.int32(60)
TrackerCollisionTrackMon.GoodPVtxMin = cms.double( 0.)
TrackerCollisionTrackMon.GoodPVtxMax = cms.double(60.)

#TrackerCollisionTrackMon.BXlumiBin = cms.int32(100) # (400)
#TrackerCollisionTrackMon.BXlumiMin = cms.double(1)  # (2000)
#TrackerCollisionTrackMon.BXlumiMax = cms.double(10) # (6000)
