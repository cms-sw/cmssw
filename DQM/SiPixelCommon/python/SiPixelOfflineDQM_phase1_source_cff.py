import FWCore.ParameterSet.Config as cms

# Pixel RawDataError Monitoring
from DQM.SiPixelMonitorRawData.SiPixelMonitorRawData_cfi import * 
SiPixelRawDataErrorSourcePhase1.saveFile = False
SiPixelRawDataErrorSourcePhase1.isPIB = False
SiPixelRawDataErrorSourcePhase1.slowDown = False
SiPixelRawDataErrorSourcePhase1.reducedSet = False

# Pixel Digi Monitoring
from DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi import *
SiPixelDigiSourcePhase1.saveFile = False
SiPixelDigiSourcePhase1.isPIB = False
SiPixelDigiSourcePhase1.slowDown = False

# Pixel Cluster Monitoring
from DQM.SiPixelMonitorCluster.SiPixelMonitorCluster_cfi import *
SiPixelClusterSourcePhase1.saveFile = False

# Pixel RecHit Monitoring
from DQM.SiPixelMonitorRecHit.SiPixelMonitorRecHit_cfi import *
SiPixelRecHitSourcePhase1.saveFile = False

# Pixel Track Monitoring
from DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_cfi import *
SiPixelTrackResidualSourcePhase1.saveFile = False
SiPixelTrackResidualSourcePhase1.TrackCandidateProducer = cms.string('initialStepTrackCandidates')
SiPixelTrackResidualSourcePhase1.trajectoryInput = cms.InputTag('generalTracks')
from DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_Cosmics_cfi import *
SiPixelTrackResidualSource_CosmicsPhase1.saveFile = False
from DQM.SiPixelMonitorTrack.SiPixelMonitorEfficiency_cfi import *
SiPixelHitEfficiencySourcePhase1.saveFile = False
SiPixelHitEfficiencySourcePhase1.trajectoryInput = cms.InputTag('generalTracks') 

##online/offline
#RawDataErrors
SiPixelRawDataErrorSourcePhase1.modOn = False
SiPixelRawDataErrorSourcePhase1.ladOn = True
SiPixelRawDataErrorSourcePhase1.bladeOn = True
#Digi
SiPixelDigiSourcePhase1.modOn = False
SiPixelDigiSourcePhase1.twoDimOn = False
SiPixelDigiSourcePhase1.reducedSet = True
SiPixelDigiSourcePhase1.hiRes = False
SiPixelDigiSourcePhase1.twoDimModOn = False
SiPixelDigiSourcePhase1.twoDimOnlyLayDisk = False
SiPixelDigiSourcePhase1.ladOn = True
SiPixelDigiSourcePhase1.layOn = True
SiPixelDigiSourcePhase1.phiOn = False
SiPixelDigiSourcePhase1.bladeOn = True
SiPixelDigiSourcePhase1.diskOn = True
SiPixelDigiSourcePhase1.ringOn = False
SiPixelDigiSourcePhase1.bigEventSize = 2600
#Cluster
SiPixelClusterSourcePhase1.modOn = False
SiPixelClusterSourcePhase1.twoDimOn = False
SiPixelClusterSourcePhase1.reducedSet = True
SiPixelClusterSourcePhase1.ladOn = True
SiPixelClusterSourcePhase1.layOn = True
SiPixelClusterSourcePhase1.phiOn = False
SiPixelClusterSourcePhase1.bladeOn = True
SiPixelClusterSourcePhase1.diskOn = True
SiPixelClusterSourcePhase1.ringOn = False
SiPixelClusterSourcePhase1.bigEventSize = 180
#RecHit
SiPixelRecHitSourcePhase1.modOn = False
SiPixelRecHitSourcePhase1.twoDimOn = False
SiPixelRecHitSourcePhase1.reducedSet = True
SiPixelRecHitSourcePhase1.ladOn = True
SiPixelRecHitSourcePhase1.layOn = True
SiPixelRecHitSourcePhase1.phiOn = False	
SiPixelRecHitSourcePhase1.bladeOn = True
SiPixelRecHitSourcePhase1.diskOn = True
SiPixelRecHitSourcePhase1.ringOn = False

#Track
SiPixelTrackResidualSourcePhase1.modOn = False
SiPixelTrackResidualSourcePhase1.ladOn = True
SiPixelTrackResidualSourcePhase1.layOn = True
SiPixelTrackResidualSourcePhase1.phiOn = False	
SiPixelTrackResidualSourcePhase1.bladeOn = True
SiPixelTrackResidualSourcePhase1.diskOn = True
SiPixelTrackResidualSourcePhase1.ringOn = False
SiPixelTrackResidualSource_CosmicsPhase1.modOn = False
SiPixelTrackResidualSource_CosmicsPhase1.ladOn = True
SiPixelTrackResidualSource_CosmicsPhase1.layOn = True
SiPixelTrackResidualSource_CosmicsPhase1.phiOn = False	
SiPixelTrackResidualSource_CosmicsPhase1.bladeOn = True
SiPixelTrackResidualSource_CosmicsPhase1.diskOn = True
SiPixelTrackResidualSource_CosmicsPhase1.ringOn = False
SiPixelHitEfficiencySourcePhase1.modOn = False
SiPixelHitEfficiencySourcePhase1.ladOn = True
SiPixelHitEfficiencySourcePhase1.layOn = False
SiPixelHitEfficiencySourcePhase1.phiOn = False
SiPixelHitEfficiencySourcePhase1.bladeOn = True
SiPixelHitEfficiencySourcePhase1.diskOn = False
SiPixelHitEfficiencySourcePhase1.ringOn = False

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
SiPixelRawDataErrorSource_phase1 = SiPixelRawDataErrorSourcePhase1.clone(
    )
SiPixelDigiSource_phase1 = SiPixelDigiSourcePhase1.clone(
    )
SiPixelClusterSource_phase1 = SiPixelClusterSourcePhase1.clone(
    )
SiPixelRecHitSource_phase1 = SiPixelRecHitSourcePhase1.clone(
    )
SiPixelTrackResidualSource_phase1 = SiPixelTrackResidualSourcePhase1.clone(
    )
SiPixelHitEfficiencySource_phase1 = SiPixelHitEfficiencySourcePhase1.clone(
    )


#DQM service
dqmInfo = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Pixel')
)

#FED integrity
from DQM.SiPixelMonitorRawData.SiPixelMonitorHLT_cfi import *

#siPixelOfflineDQM_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource + SiPixelHitEfficiencySource + dqmInfo)

#siPixelOfflineDQM_cosmics_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource_Cosmics + dqmInfo)

#siPixelOfflineDQM_heavyions_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource_HeavyIons + SiPixelHitEfficiencySource_HeavyIons + dqmInfo)

#siPixelOfflineDQM_source_woTrack = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + dqmInfo)

#siPixelOfflineDQM_phase1_source = cms.Sequence(SiPixelRawDataErrorSource_phase1 + SiPixelDigiSource_phase1 + SiPixelRecHitSource_phase1 + SiPixelClusterSource_phase1 + SiPixelTrackResidualSource_phase1 + SiPixelHitEfficiencySource_phase1 + dqmInfo)
