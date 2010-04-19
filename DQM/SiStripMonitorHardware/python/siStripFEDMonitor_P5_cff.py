import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorHardware.siStripFEDMonitor_cfi import *

#disable error output: enabled by default.
siStripFEDMonitor.PrintDebugMessages = 0
#lumi histogram
siStripFEDMonitor.ErrorFractionByLumiBlockHistogramConfig.Enabled = False
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
siStripFEDMonitor.FETimeDiffTIBHistogramConfig.Enabled = True
siStripFEDMonitor.FETimeDiffTOBHistogramConfig.Enabled = True
siStripFEDMonitor.FETimeDiffTECBHistogramConfig.Enabled = True
siStripFEDMonitor.FETimeDiffTECFHistogramConfig.Enabled = True
siStripFEDMonitor.ApveAddressHistogramConfig.Enabled = True
siStripFEDMonitor.FeMajAddressHistogramConfig.Enabled = True
#medians per APV for all channels, all events
siStripFEDMonitor.MedianAPV0HistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(256),
  Min = cms.double(0),
  Max = cms.double(1024)
  )
siStripFEDMonitor.MedianAPV1HistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(256),
  Min = cms.double(0),
  Max = cms.double(1024)
  )      
#Error counting histograms
siStripFEDMonitor.nFEDErrorsHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(101),
  Min = cms.double(0),
  Max = cms.double(101)
)
siStripFEDMonitor.nFEDDAQProblemsHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(101),
  Min = cms.double(0),
  Max = cms.double(101)
)
siStripFEDMonitor.nFEDsWithFEProblemsHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(101),
  Min = cms.double(0),
  Max = cms.double(101)
)
siStripFEDMonitor.nFEDCorruptBuffersHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(101),
  Min = cms.double(0),
  Max = cms.double(101)
)
#bins size number of FE Units/10, max is n channels
siStripFEDMonitor.nBadChannelStatusBitsHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(250),
  Min = cms.double(0),
  Max = cms.double(500)
)
siStripFEDMonitor.nBadActiveChannelStatusBitsHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(250),
  Min = cms.double(0),
  Max = cms.double(500)
)
siStripFEDMonitor.nFEDsWithFEOverflowsHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(101),
  Min = cms.double(0),
  Max = cms.double(101)
)
siStripFEDMonitor.nFEDsWithMissingFEsHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(101),
  Min = cms.double(0),
  Max = cms.double(101)
)
siStripFEDMonitor.nFEDsWithFEBadMajorityAddressesHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(101),
  Min = cms.double(0),
  Max = cms.double(101)
)
siStripFEDMonitor.nUnconnectedChannelsHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(250),
  Min = cms.double(0),
  Max = cms.double(500)
)
siStripFEDMonitor.nAPVStatusBitHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(250),
  #Min = cms.double(0),
  #Max = cms.double(500)
)
siStripFEDMonitor.nAPVErrorHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(250),
  Min = cms.double(0),
  Max = cms.double(500)
)
siStripFEDMonitor.nAPVAddressErrorHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(250),
  Min = cms.double(0),
  Max = cms.double(500)
)
siStripFEDMonitor.nUnlockedHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(250),
  Min = cms.double(0),
  Max = cms.double(500)
)
siStripFEDMonitor.nOutOfSyncHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(250),
  Min = cms.double(0),
  Max = cms.double(500)
)
siStripFEDMonitor.nTotalBadChannelsHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(250),
  #Min = cms.double(0),
  #Max = cms.double(500)
)
siStripFEDMonitor.nTotalBadActiveChannelsHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(250),
  #Min = cms.double(0),
  #Max = cms.double(500)
)
siStripFEDMonitor.nTotalBadChannelsvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(600),
  Min = cms.double(0),
  Max = cms.double(3600)
)
siStripFEDMonitor.nTotalBadActiveChannelsvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(600),
  Min = cms.double(0),
  Max = cms.double(3600)
)
siStripFEDMonitor.nFEDErrorsvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(600),
  Min = cms.double(0),
  Max = cms.double(3600)
)
siStripFEDMonitor.nFEDCorruptBuffersvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(600),
  #Min = cms.double(0),
  #Max = cms.double(3600)
)
siStripFEDMonitor.nFEDsWithFEProblemsvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(600),
  #Min = cms.double(0),
  #Max = cms.double(3600)
)
siStripFEDMonitor.nAPVStatusBitvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(600),
  #Min = cms.double(0),
  #Max = cms.double(3600)
)
siStripFEDMonitor.nAPVErrorvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(600),
  Min = cms.double(0),
  Max = cms.double(3600)
)
siStripFEDMonitor.nAPVAddressErrorvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(600),
  Min = cms.double(0),
  Max = cms.double(3600)
)
siStripFEDMonitor.nUnlockedvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(600),
  Min = cms.double(0),
  Max = cms.double(3600)
)
siStripFEDMonitor.nOutOfSyncvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(600),
  Min = cms.double(0),
  Max = cms.double(3600)
)
