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
from DQM.SiPixelMonitorTrack.RefitterForPixelDQM import *
from DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_cfi import *
SiPixelTrackResidualSource.saveFile = False
from DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_Cosmics_cfi import *
SiPixelTrackResidualSource_Cosmics.saveFile = False
from DQM.SiPixelMonitorTrack.SiPixelMonitorEfficiency_cfi import *
SiPixelHitEfficiencySource.saveFile = False
from DQM.TrackerMonitorTrack.SiPixelMonitorTrackResiduals_cfi import *

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
hiTracks = "hiGeneralTracks"
hiRefittedForPixelDQM= refittedForPixelDQM.clone()
hiRefittedForPixelDQM.src=hiTracks

SiPixelTrackResidualSource_HeavyIons = SiPixelTrackResidualSource.clone(
    vtxsrc='hiSelectedVertex'
    )

SiPixelHitEfficiencySource_HeavyIons = SiPixelHitEfficiencySource.clone(
    vtxsrc='hiSelectedVertex'
    )


#DQM service
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfo = DQMEDAnalyzer('DQMEventInfo',
    subSystemFolder = cms.untracked.string('Pixel')
)

#Check if perLSsaving is enabled to mask MEs vs LS
from DQMServices.Core.DQMStore_cfi import DQMStore
if(DQMStore.saveByLumi):
    SiPixelDigiSource.perLSsaving=True

#FED integrity
from DQM.SiPixelMonitorRawData.SiPixelMonitorHLT_cfi import *

siPixelOfflineDQM_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + refittedForPixelDQM + SiPixelTrackResidualSource + SiPixelHitEfficiencySource + SiPixelMonitorTrackResiduals + dqmInfo)

siPixelOfflineDQM_cosmics_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + SiPixelTrackResidualSource_Cosmics + dqmInfo)

siPixelOfflineDQM_heavyions_source = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + hiRefittedForPixelDQM + SiPixelTrackResidualSource_HeavyIons + SiPixelHitEfficiencySource_HeavyIons + dqmInfo)

siPixelOfflineDQM_source_woTrack = cms.Sequence(SiPixelHLTSource + SiPixelRawDataErrorSource + SiPixelDigiSource + SiPixelRecHitSource + SiPixelClusterSource + dqmInfo)

# Phase1 config
# _all_ of the stuff above becomes obsolete. We just hijack the names and 
# replace them with the phase1 config of the new DQM.
from DQM.SiPixelPhase1Config.SiPixelPhase1OfflineDQM_source_cff import *
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toReplaceWith(siPixelOfflineDQM_source, siPixelPhase1OfflineDQM_source)
phase1Pixel.toReplaceWith(siPixelOfflineDQM_cosmics_source, siPixelPhase1OfflineDQM_source_cosmics)
phase1Pixel.toReplaceWith(siPixelOfflineDQM_heavyions_source, siPixelPhase1OfflineDQM_source_hi)
# don't forget the Harvesters, they are plugged in at PixelOfflineDQMClient
# TODO: the same game for the other three.
