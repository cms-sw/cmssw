import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorHardware.siStripFEDMonitor_cfi import *

#disable error output: only in P5 configuration.
siStripFEDMonitor.PrintDebugMessages = 0
#Global/summary histograms
siStripFEDMonitor.DataPresentHistogramConfig.Enabled = True
siStripFEDMonitor.AnyFEDErrorsHistogramConfig.Enabled = True
siStripFEDMonitor.AnyDAQProblemsHistogramConfig.Enabled = True
siStripFEDMonitor.AnyFEProblemsHistogramConfig.Enabled = True
siStripFEDMonitor.CorruptBuffersHistogramConfig.Enabled = True
siStripFEDMonitor.BadChannelStatusBitsHistogramConfig.Enabled = False
siStripFEDMonitor.BadActiveChannelStatusBitsHistogramConfig.Enabled = True
#sub sets of FE problems
siStripFEDMonitor.FEOverflowsHistogramConfig.Enabled = False
siStripFEDMonitor.FEMissingHistogramConfig.Enabled = False
siStripFEDMonitor.BadMajorityAddressesHistogramConfig.Enabled = False
#Sub sets of DAQ problems
siStripFEDMonitor.DataMissingHistogramConfig.Enabled = False
siStripFEDMonitor.BadIDsHistogramConfig.Enabled = False
siStripFEDMonitor.BadDAQPacketHistogramConfig.Enabled = False
siStripFEDMonitor.InvalidBuffersHistogramConfig.Enabled = False
siStripFEDMonitor.BadDAQCRCsHistogramConfig.Enabled = False
siStripFEDMonitor.BadFEDCRCsHistogramConfig.Enabled = False
#TkHistoMap
siStripFEDMonitor.TkHistoMapHistogramConfig.Enabled = True
#Detailed FED level expert histograms
siStripFEDMonitor.FEOverflowsDetailedHistogramConfig.Enabled = False
siStripFEDMonitor.FEMissingDetailedHistogramConfig.Enabled = False
siStripFEDMonitor.BadMajorityAddressesDetailedHistogramConfig.Enabled = False
siStripFEDMonitor.BadAPVStatusBitsDetailedHistogramConfig.Enabled = False
siStripFEDMonitor.APVErrorBitsDetailedHistogramConfig.Enabled = False
siStripFEDMonitor.APVAddressErrorBitsDetailedHistogramConfig.Enabled = False
siStripFEDMonitor.UnlockedBitsDetailedHistogramConfig.Enabled = False
siStripFEDMonitor.OOSBitsDetailedHistogramConfig.Enabled = False
#Error counting histograms
siStripFEDMonitor.nFEDErrorsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(441),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(441)
)
siStripFEDMonitor.nFEDDAQProblemsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(441),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(441)
)
siStripFEDMonitor.nFEDsWithFEProblemsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(441),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(441)
)
siStripFEDMonitor.nFEDCorruptBuffersHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(441),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(441)
)
#bins size number of FE Units/10, max is n channels
siStripFEDMonitor.nBadChannelStatusBitsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(353),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(42241)
)
siStripFEDMonitor.nBadActiveChannelStatusBitsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(353),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(42241)
)
siStripFEDMonitor.nFEDsWithFEOverflowsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(441),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(441)
)
siStripFEDMonitor.nFEDsWithMissingFEsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(441),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(441)
)
siStripFEDMonitor.nFEDsWithFEBadMajorityAddressesHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(441),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(441)
)
siStripFEDMonitor.nTotalBadChannelsvsEvtNumHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(1000),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(1000)
)
siStripFEDMonitor.nTotalBadActiveChannelsvsEvtNumHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(1000),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(1000)
)
