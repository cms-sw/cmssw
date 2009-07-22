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
from DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_Cosmics_cfi import *
SiPixelTrackResidualSource_Cosmics.saveFile = False

##online/offline
#RawDataErrors
SiPixelRawDataErrorSource.modOn = False
SiPixelRawDataErrorSource.ladOn = True
SiPixelRawDataErrorSource.layOn = False
SiPixelRawDataErrorSource.phiOn = False
SiPixelRawDataErrorSource.bladeOn = True
SiPixelRawDataErrorSource.diskOn = False
SiPixelRawDataErrorSource.ringOn = False
#Digi
SiPixelDigiSource.modOn = False
SiPixelDigiSource.twoDimOn = False
SiPixelDigiSource.reducedSet = True
SiPixelDigiSource.hiRes = False
SiPixelDigiSource.ladOn = True
SiPixelDigiSource.layOn = True
SiPixelDigiSource.phiOn = False
SiPixelDigiSource.bladeOn = True
SiPixelDigiSource.diskOn = True
SiPixelDigiSource.ringOn = False
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

#DQM service
dqmInfo = cms.EDFilter("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Pixel')
)

#FED integrity
from DQM.SiPixelMonitorRawData.SiPixelMonitorHLT_cfi import *

siPixelOfflineDQM_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource + dqmInfo)

siPixelOfflineDQM_cosmics_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource_Cosmics + dqmInfo)

siPixelOfflineDQM_source_woTrack = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + dqmInfo)
