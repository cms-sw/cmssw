import FWCore.ParameterSet.Config as cms

from DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi import *
from DQM.SiPixelMonitorCluster.SiPixelMonitorCluster_cfi import *
from DQM.SiPixelMonitorRecHit.SiPixelMonitorRecHit_cfi import *
siPixelOfflineDQM_source = cms.Sequence(SiPixelDigiSource*SiPixelClusterSource*SiPixelRecHitSource)
SiPixelDigiSource.saveFile = False
SiPixelClusterSource.saveFile = False
SiPixelRecHitSource.saveFile = False
SiPixelDigiSource.modOn = False
SiPixelDigiSource.ladOn = True
SiPixelDigiSource.layOn = True
SiPixelDigiSource.phiOn = True
SiPixelDigiSource.bladeOn = True
SiPixelDigiSource.diskOn = True
SiPixelDigiSource.ringOn = True
SiPixelClusterSource.modOn = False
SiPixelClusterSource.ladOn = True
SiPixelClusterSource.layOn = True
SiPixelClusterSource.phiOn = True
SiPixelClusterSource.bladeOn = True
SiPixelClusterSource.diskOn = True
SiPixelClusterSource.ringOn = True
SiPixelRecHitSource.modOn = False
SiPixelRecHitSource.ladOn = True
SiPixelRecHitSource.layOn = True
SiPixelRecHitSource.phiOn = True
SiPixelRecHitSource.bladeOn = True
SiPixelRecHitSource.diskOn = True
SiPixelRecHitSource.ringOn = True

