import FWCore.ParameterSet.Config as cms

# Pixel Digi Monitoring
from DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi import *
siPixelOfflineDQM_source = cms.Sequence(SiPixelDigiSource)
SiPixelDigiSource.saveFile = False
SiPixelDigiSource.isPIB = True
SiPixelDigiSource.slowDown = False
#online/offline
SiPixelDigiSource.modOn = True
SiPixelDigiSource.ladOn = True
SiPixelDigiSource.layOn = True
SiPixelDigiSource.phiOn = True
SiPixelDigiSource.bladeOn = True
SiPixelDigiSource.diskOn = True
SiPixelDigiSource.ringOn = True

