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
SiPixelDigiSource.hiRes = True
SiPixelDigiSource.reducedSet = False
SiPixelDigiSource.twoDimModOn = False
SiPixelDigiSource.twoDimOnlyLayDisk = True

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
SiPixelTrackResidualSource.TrackCandidateProducer = cms.string('newTrackCandidateMaker')
SiPixelTrackResidualSource.trajectoryInput = cms.InputTag('generalTracks')
from DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_Cosmics_cfi import *
SiPixelTrackResidualSource_Cosmics.saveFile = False
SiPixelTrackResidualSource_Cosmics.TrackCandidateProducer = cms.string('ckfTrackCandidatesP5')
SiPixelTrackResidualSource_Cosmics.trajectoryInput = cms.string('ctfWithMaterialTracksP5')
from DQM.SiPixelMonitorTrack.SiPixelMonitorEfficiency_cfi import *
SiPixelHitEfficiencySource.saveFile = False
SiPixelHitEfficiencySource.trajectoryInput = cms.InputTag('generalTracks') 
from DQM.TrackerMonitorTrack.SiPixelMonitorTrackResiduals_cfi import *

##online/offline
#RawDataErrors
SiPixelRawDataErrorSource.modOn = True
SiPixelRawDataErrorSource.ladOn = False
SiPixelRawDataErrorSource.bladeOn = False
#Digi
SiPixelDigiSource.modOn = True
SiPixelDigiSource.twoDimOn = True
SiPixelDigiSource.reducedSet = False
SiPixelDigiSource.hiRes = True ## do not set to False, otherwise occupancy map code will crash!
SiPixelDigiSource.ladOn = False
SiPixelDigiSource.layOn = False
SiPixelDigiSource.phiOn = False
SiPixelDigiSource.bladeOn = False
SiPixelDigiSource.diskOn = False
SiPixelDigiSource.ringOn = False
SiPixelDigiSource.bigEventSize = 5000
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
SiPixelClusterSource.bigEventSize = 330
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
SiPixelHitEfficiencySource.modOn = True
SiPixelHitEfficiencySource.ladOn = False
SiPixelHitEfficiencySource.layOn = False
SiPixelHitEfficiencySource.phiOn = False
SiPixelHitEfficiencySource.bladeOn = False
SiPixelHitEfficiencySource.diskOn = False
SiPixelHitEfficiencySource.ringOn = False

#HI track modules
hiTracks = "hiGlobalPrimTracks"

SiPixelTrackResidualSource_HeavyIons = SiPixelTrackResidualSource.clone(
    TrackCandidateProducer = 'hiPrimTrackCandidates',
    trajectoryInput = hiTracks,
    vtxsrc='hiSelectedVertex'
    )

SiPixelHitEfficiencySource_HeavyIons = SiPixelHitEfficiencySource.clone(
    trajectoryInput = hiTracks,
    vtxsrc='hiSelectedVertex'
    )

#DQM service
dqmInfo = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Pixel')
)

#FED integrity
from DQM.SiPixelMonitorRawData.SiPixelMonitorHLT_cfi import *
SiPixelHLTSource.DirName = cms.untracked.string('Pixel/FEDIntegrity/')

siPixelP5DQM_source = cms.Sequence(SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource + SiPixelHitEfficiencySource + SiPixelMonitorTrackResiduals + dqmInfo)

siPixelP5DQM_cosmics_source = cms.Sequence(SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource_Cosmics + dqmInfo)

siPixelP5DQM_heavyions_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource_HeavyIons + SiPixelHitEfficiencySource_HeavyIons + dqmInfo)

siPixelP5DQM_source_woTrack = cms.Sequence(SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + dqmInfo)
