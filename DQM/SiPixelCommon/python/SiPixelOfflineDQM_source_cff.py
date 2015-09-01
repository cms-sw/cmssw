import FWCore.ParameterSet.Config as cms

# Pixel RawDataError Monitoring
from DQM.SiPixelMonitorRawData.SiPixelMonitorRawData_cfi import * 
SiPixelRawDataErrorSource.saveFile = False
SiPixelRawDataErrorSource.isPIB = False
SiPixelRawDataErrorSource.slowDown = False
SiPixelRawDataErrorSource.reducedSet = False

# Pixel Digi Monitoring
from DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi import *
SiPixelDigiSource.saveFile = False
SiPixelDigiSource.isPIB = False
SiPixelDigiSource.slowDown = False

# Pixel Cluster Monitoring
from DQM.SiPixelMonitorCluster.SiPixelMonitorCluster_cfi import *
SiPixelClusterSource.saveFile = False

# Pixel RecHit Monitoring
from DQM.SiPixelMonitorRecHit.SiPixelMonitorRecHit_cfi import *
SiPixelRecHitSource.saveFile = False

# Pixel Track Monitoring
from DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_cfi import *
SiPixelTrackResidualSource.saveFile = False
SiPixelTrackResidualSource.TrackCandidateProducer = cms.string('initialStepTrackCandidates')
SiPixelTrackResidualSource.trajectoryInput = cms.InputTag('generalTracks')
from DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_Cosmics_cfi import *
SiPixelTrackResidualSource_Cosmics.saveFile = False
from DQM.SiPixelMonitorTrack.SiPixelMonitorEfficiency_cfi import *
SiPixelHitEfficiencySource.saveFile = False
SiPixelHitEfficiencySource.trajectoryInput = cms.InputTag('generalTracks') 

##online/offline
#RawDataErrors
SiPixelRawDataErrorSource.modOn = False
SiPixelRawDataErrorSource.ladOn = True
SiPixelRawDataErrorSource.bladeOn = True
#Digi
SiPixelDigiSource.modOn = False
SiPixelDigiSource.twoDimOn = False
SiPixelDigiSource.reducedSet = True
SiPixelDigiSource.hiRes = False
SiPixelDigiSource.twoDimModOn = False
SiPixelDigiSource.twoDimOnlyLayDisk = False
SiPixelDigiSource.ladOn = True
SiPixelDigiSource.layOn = True
SiPixelDigiSource.phiOn = False
SiPixelDigiSource.bladeOn = True
SiPixelDigiSource.diskOn = True
SiPixelDigiSource.ringOn = False
SiPixelDigiSource.bigEventSize = 2600
SiPixelDigiSource.isOffline = True
#Cluster
SiPixelClusterSource.modOn = False
SiPixelClusterSource.twoDimOn = False
SiPixelClusterSource.reducedSet = True
SiPixelClusterSource.ladOn = True
SiPixelClusterSource.layOn = True
SiPixelClusterSource.phiOn = False
SiPixelClusterSource.bladeOn = True
SiPixelClusterSource.diskOn = True
SiPixelClusterSource.ringOn = False
SiPixelClusterSource.bigEventSize = 180
#RecHit
SiPixelRecHitSource.modOn = False
SiPixelRecHitSource.twoDimOn = False
SiPixelRecHitSource.reducedSet = True
SiPixelRecHitSource.ladOn = True
SiPixelRecHitSource.layOn = True
SiPixelRecHitSource.phiOn = False	
SiPixelRecHitSource.bladeOn = True
SiPixelRecHitSource.diskOn = True
SiPixelRecHitSource.ringOn = False

#Track
SiPixelTrackResidualSource.modOn = False
SiPixelTrackResidualSource.ladOn = True
SiPixelTrackResidualSource.layOn = True
SiPixelTrackResidualSource.phiOn = False	
SiPixelTrackResidualSource.bladeOn = True
SiPixelTrackResidualSource.diskOn = True
SiPixelTrackResidualSource.ringOn = False
SiPixelTrackResidualSource_Cosmics.modOn = False
SiPixelTrackResidualSource_Cosmics.ladOn = True
SiPixelTrackResidualSource_Cosmics.layOn = True
SiPixelTrackResidualSource_Cosmics.phiOn = False	
SiPixelTrackResidualSource_Cosmics.bladeOn = True
SiPixelTrackResidualSource_Cosmics.diskOn = True
SiPixelTrackResidualSource_Cosmics.ringOn = False
SiPixelHitEfficiencySource.modOn = False
SiPixelHitEfficiencySource.ladOn = True
SiPixelHitEfficiencySource.layOn = False
SiPixelHitEfficiencySource.phiOn = False
SiPixelHitEfficiencySource.bladeOn = True
SiPixelHitEfficiencySource.diskOn = False
SiPixelHitEfficiencySource.ringOn = False

#HI track modules
hiTracks = "hiGlobalPrimTracks"

SiPixelTrackResidualSource_HeavyIons = SiPixelTrackResidualSource.clone(
    TrackCandidateProducer = 'hiPrimTrackCandidates',
    trajectoryInput = hiTracks
    )

SiPixelHitEfficiencySource_HeavyIons = SiPixelHitEfficiencySource.clone(
    trajectoryInput = hiTracks
    )


# Phase1 Upgrade configuration
SiPixelRawDataErrorSource_phase1 = SiPixelRawDataErrorSource.clone(
    isUpgrade = cms.untracked.bool(True)
    )
SiPixelDigiSource_phase1 = SiPixelDigiSource.clone(
    isUpgrade = cms.untracked.bool(True)
    )
SiPixelClusterSource_phase1 = SiPixelClusterSource.clone(
    isUpgrade = cms.untracked.bool(True)
    )
SiPixelRecHitSource_phase1 = SiPixelRecHitSource.clone(
    isUpgrade = cms.untracked.bool(True)
    )
SiPixelTrackResidualSource_phase1 = SiPixelTrackResidualSource.clone(
    isUpgrade = cms.untracked.bool(True)
    )
SiPixelHitEfficiencySource_phase1 = SiPixelHitEfficiencySource.clone(
    isUpgrade = cms.untracked.bool(True)
    )


#DQM service
dqmInfo = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Pixel')
)

#FED integrity
from DQM.SiPixelMonitorRawData.SiPixelMonitorHLT_cfi import *

siPixelOfflineDQM_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource + SiPixelHitEfficiencySource + dqmInfo)

siPixelOfflineDQM_cosmics_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource_Cosmics + dqmInfo)

siPixelOfflineDQM_heavyions_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource_HeavyIons + SiPixelHitEfficiencySource_HeavyIons + dqmInfo)

siPixelOfflineDQM_source_woTrack = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + dqmInfo)

siPixelOfflineDQM_phase1_source = cms.Sequence(SiPixelRawDataErrorSource_phase1 + SiPixelDigiSource_phase1 + SiPixelRecHitSource_phase1 + SiPixelClusterSource_phase1 + SiPixelTrackResidualSource_phase1 + SiPixelHitEfficiencySource_phase1 + dqmInfo)
