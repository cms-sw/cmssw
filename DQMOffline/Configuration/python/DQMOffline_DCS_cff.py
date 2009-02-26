import FWCore.ParameterSet.Config as cms

siStripDcsInfo = cms.EDFilter("SiStripDcsInfo")
siPixelDcsInfo = cms.EDFilter("SiPixelDcsInfo")
from DQM.EcalBarrelMonitorTasks.EBDcsInfoTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EEDcsInfoTask_cfi import *

dcs_dqmoffline = cms.Sequence(siStripDcsInfo*siPixelDcsInfo*ecalBarrelDcsInfoTask*ecalEndcapDcsInfoTask)

