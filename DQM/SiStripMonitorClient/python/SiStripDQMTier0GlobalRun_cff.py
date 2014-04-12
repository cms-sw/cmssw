import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# DQM Services
#-------------------------------------------------
# core
from DQMServices.Core.DQM_cfg import *
# ME2EDM conversion
from DQMServices.Components.MEtoEDMConverter_cfi import *
#-------------------------------------------------
# SiStrip DQM Source and Client
#-------------------------------------------------
# source
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_Cosmic_cff import *
#-------------------------------------------------
# Scheduling
#-------------------------------------------------
SiStripDQMTest_cosmicTk = cms.Sequence(SiStripDQMTier0_cosmicTk*MEtoEDMConverter)
SiStripDQMTest_ckf = cms.Sequence(SiStripDQMTier0_ckf*MEtoEDMConverter)
#SiStripDQMTest_rs = cms.Sequence(SiStripDQMTier0_rs*MEtoEDMConverter)
SiStripDQMTest = cms.Sequence(SiStripDQMTier0*MEtoEDMConverter)

