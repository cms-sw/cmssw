import FWCore.ParameterSet.Config as cms

# SiStripMonitorDigi
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi import *

SiStripMonitorSimDigi = SiStripMonitorDigi.clone()
SiStripMonitorSimDigi.digiTopDir = cms.string("SiStripVal")
SiStripMonitorSimDigi.DigiProducersList = cms.VInputTag(
    cms.InputTag("simSiStripDigis","ZeroSuppressed")
)    

SiStripMonitorSimDigi.Mod_On = False
SiStripMonitorSimDigi.TProfDigiApvCycle.subdetswitchon = True

# APV shots monitoring
SiStripMonitorSimDigi.TkHistoMapNApvShots_On = True 
SiStripMonitorSimDigi.TkHistoMapNStripApvShots_On= True
SiStripMonitorSimDigi.TkHistoMapMedianChargeApvShots_On= True
SiStripMonitorSimDigi.TH1NApvShots.subdetswitchon = True
SiStripMonitorSimDigi.TH1NApvShots.globalswitchon = True
SiStripMonitorSimDigi.TH1ChargeMedianApvShots.subdetswitchon = True
SiStripMonitorSimDigi.TH1ChargeMedianApvShots.globalswitchon = True
SiStripMonitorSimDigi.TH1NStripsApvShots.subdetswitchon = True
SiStripMonitorSimDigi.TH1NStripsApvShots.globalswitchon = True
SiStripMonitorSimDigi.TH1ApvNumApvShots.subdetswitchon = True
SiStripMonitorSimDigi.TH1ApvNumApvShots.globalswitchon = True
SiStripMonitorSimDigi.TProfNShotsVsTime.subdetswitchon = True
SiStripMonitorSimDigi.TProfNShotsVsTime.globalswitchon = True
SiStripMonitorSimDigi.TProfGlobalNShots.globalswitchon = True

SiStripMonitorSimDigi.TH1NumberOfDigisPerStrip.moduleswitchon = cms.bool(True)
SiStripMonitorSimDigi.TProfNumberOfDigi.layerswitchon = cms.bool(True)
SiStripMonitorSimDigi.TProfNumberOfDigi.moduleswitchon = cms.bool(True)
SiStripMonitorSimDigi.TProfTotalNumberOfDigis.subdetswitchon = cms.bool(True)
SiStripMonitorSimDigi.TH2DigiApvCycle.subdetswitchon = cms.bool(True)


