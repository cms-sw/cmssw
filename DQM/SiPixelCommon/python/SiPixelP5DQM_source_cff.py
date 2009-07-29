import FWCore.ParameterSet.Config as cms

# Pixel RawDataError Monitoring
from DQM.SiPixelMonitorRawData.SiPixelMonitorRawData_cfi import * 
SiPixelRawDataErrorSource.saveFile = False
SiPixelRawDataErrorSource.isPIB = False
SiPixelRawDataErrorSource.slowDown = False
SiPixelRawDataErrorSource.reducedSet = True

# Pixel Digi Monitoring
from DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi import *
SiPixelDigiSource.saveFile = False
SiPixelDigiSource.isPIB = False
SiPixelDigiSource.slowDown = False

# Pixel Cluster Monitoring
from DQM.SiPixelMonitorCluster.SiPixelMonitorCluster_cfi import *
SiPixelClusterSource.saveFile = False
SiPixelClusterSource.isPIB = False

# Pixel RecHit Monitoring
from DQM.SiPixelMonitorRecHit.SiPixelMonitorRecHit_cfi import *
SiPixelRecHitSource.saveFile = False
SiPixelRecHitSource.isPIB = False

# Pixel Track Monitoring
from DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_cfi import *
SiPixelTrackResidualSource.saveFile = False
from DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_Cosmics_cfi import *
SiPixelTrackResidualSource_Cosmics.saveFile = False
#MC
SiPixelTrackResidualSource.TrackCandidateProducer = cms.string('newTrackCandidateMaker')
SiPixelTrackResidualSource.trajectoryInput = cms.InputTag('generalTracks')

##online/offline
#RawDataErrors
SiPixelRawDataErrorSource.modOn = True
SiPixelRawDataErrorSource.ladOn = False
SiPixelRawDataErrorSource.layOn = False
SiPixelRawDataErrorSource.phiOn = False
SiPixelRawDataErrorSource.bladeOn = False
SiPixelRawDataErrorSource.diskOn = False
SiPixelRawDataErrorSource.ringOn = False
#Digi
SiPixelDigiSource.modOn = True
SiPixelDigiSource.twoDimOn = True
SiPixelDigiSource.reducedSet = False
SiPixelDigiSource.hiRes = False
SiPixelDigiSource.ladOn = False
SiPixelDigiSource.layOn = False
SiPixelDigiSource.phiOn = False
SiPixelDigiSource.bladeOn = False
SiPixelDigiSource.diskOn = False
SiPixelDigiSource.ringOn = False
#Cluster
SiPixelClusterSource.modOn = True
SiPixelClusterSource.twoDimOn = True
SiPixelClusterSource.reducedSet = True
SiPixelClusterSource.ladOn = False
SiPixelClusterSource.layOn = False
SiPixelClusterSource.phiOn = False
SiPixelClusterSource.bladeOn = False
SiPixelClusterSource.diskOn = False
SiPixelClusterSource.ringOn = False
#RecHit
SiPixelRecHitSource.modOn = True
SiPixelRecHitSource.twoDimOn = True
SiPixelRecHitSource.reducedSet = True
SiPixelRecHitSource.ladOn = False
SiPixelRecHitSource.layOn = False
SiPixelRecHitSource.phiOn = False	
SiPixelRecHitSource.bladeOn = False
SiPixelRecHitSource.diskOn = False
SiPixelRecHitSource.ringOn = False

#Track
SiPixelTrackResidualSource.modOn = True
SiPixelTrackResidualSource.ladOn = False
SiPixelTrackResidualSource.layOn = False
SiPixelTrackResidualSource.phiOn = False	
SiPixelTrackResidualSource.bladeOn = False
SiPixelTrackResidualSource.diskOn = False
SiPixelTrackResidualSource.ringOn = False
SiPixelTrackResidualSource_Cosmics.modOn = True
SiPixelTrackResidualSource_Cosmics.ladOn = False
SiPixelTrackResidualSource_Cosmics.layOn = False
SiPixelTrackResidualSource_Cosmics.phiOn = False	
SiPixelTrackResidualSource_Cosmics.bladeOn = False
SiPixelTrackResidualSource_Cosmics.diskOn = False
SiPixelTrackResidualSource_Cosmics.ringOn = False

#DQM service
dqmInfo = cms.EDFilter("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Pixel')
)

#FED integrity
from DQM.SiPixelMonitorRawData.SiPixelMonitorHLT_cfi import *

siPixelP5DQM_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource + dqmInfo)

siPixelP5DQM_cosmics_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource_Cosmics + dqmInfo)

siPixelP5DQM_source_woTrack = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + dqmInfo)
