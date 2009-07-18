import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorHardware.siStripFEDMonitor_cfi import *

#Global/summary histograms
siStripFEDMonitor.DataPresentHistogramConfig.Enabled = True
siStripFEDMonitor.AnyFEDErrorsHistogramConfig.Enabled = True
siStripFEDMonitor.AnyDAQProblemsHistogramConfig.Enabled = True
siStripFEDMonitor.AnyFEProblemsHistogramConfig.Enabled = True
siStripFEDMonitor.CorruptBuffersHistogramConfig.Enabled = True
siStripFEDMonitor.BadChannelStatusBitsHistogramConfig.Enabled = True
siStripFEDMonitor.BadActiveChannelStatusBitsHistogramConfig.Enabled = True
#Sub sets of FE problems
siStripFEDMonitor.FEOverflowsHistogramConfig.Enabled = True
siStripFEDMonitor.FEMissingHistogramConfig.Enabled = True
siStripFEDMonitor.BadMajorityAddressesHistogramConfig.Enabled = True
#Sub sets of DAQ problems
siStripFEDMonitor.DataMissingHistogramConfig.Enabled = True
siStripFEDMonitor.BadIDsHistogramConfig.Enabled = True
siStripFEDMonitor.BadDAQPacketHistogramConfig.Enabled = True
siStripFEDMonitor.InvalidBuffersHistogramConfig.Enabled = True
siStripFEDMonitor.BadDAQCRCsHistogramConfig.Enabled = True
siStripFEDMonitor.BadFEDCRCsHistogramConfig.Enabled = True
#TkHistoMap
siStripFEDMonitor.TkHistoMapHistogramConfig.Enabled = True
#Detailed FED level expert histograms
siStripFEDMonitor.FEOverflowsDetailedHistogramConfig.Enabled = True
siStripFEDMonitor.FEMissingDetailedHistogramConfig.Enabled = True
siStripFEDMonitor.BadMajorityAddressesDetailedHistogramConfig.Enabled = True
siStripFEDMonitor.BadAPVStatusBitsDetailedHistogramConfig.Enabled = True
siStripFEDMonitor.APVErrorBitsDetailedHistogramConfig.Enabled = True
siStripFEDMonitor.APVAddressErrorBitsDetailedHistogramConfig.Enabled = True
siStripFEDMonitor.UnlockedBitsDetailedHistogramConfig.Enabled = True
siStripFEDMonitor.OOSBitsDetailedHistogramConfig.Enabled = True
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
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(441),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(441)
)
siStripFEDMonitor.nFEDsWithMissingFEsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(441),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(441)
)
siStripFEDMonitor.nFEDsWithFEBadMajorityAddressesHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(441),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(441)
)
siStripFEDMonitor.nUnconnectedChannelsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(353),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(42241)
)
siStripFEDMonitor.nAPVStatusBitHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(353),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(42241)
)
siStripFEDMonitor.nAPVErrorHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(353),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(42241)
)
siStripFEDMonitor.nAPVAddressErrorHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(353),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(42241)
)
siStripFEDMonitor.nUnlockedHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(353),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(42241)
)
siStripFEDMonitor.nOutOfSyncHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(353),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(42241)
)
siStripFEDMonitor.nTotalBadChannelsvsTimeHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(600),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(3600)
)
siStripFEDMonitor.nTotalBadActiveChannelsvsTimeHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(600),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(3600)
)
siStripFEDMonitor.nFEDErrorsvsTimeHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(600),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(3600)
)
siStripFEDMonitor.nFEDCorruptBuffersvsTimeHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(600),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(3600)
)
siStripFEDMonitor.nFEDsWithFEProblemsvsTimeHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(600),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(3600)
)
siStripFEDMonitor.nAPVStatusBitvsTimeHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(600),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(3600)
)
siStripFEDMonitor.nAPVErrorvsTimeHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(600),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(3600)
)
siStripFEDMonitor.nAPVAddressErrorvsTimeHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(600),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(3600)
)
siStripFEDMonitor.nUnlockedvsTimeHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(600),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(3600)
)
siStripFEDMonitor.nOutOfSyncvsTimeHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(600),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(3600)
)
