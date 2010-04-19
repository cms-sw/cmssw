import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorHardware.siStripFEDMonitor_cfi import *

#disable error output: enabled in P5 configuration for errors.
siStripFEDMonitor.PrintDebugMessages = 0
#lumi histogram
siStripFEDMonitor.ErrorFractionByLumiBlockHistogramConfig.Enabled = True
#Global/summary histograms
siStripFEDMonitor.DataPresentHistogramConfig.Enabled = True
siStripFEDMonitor.AnyFEDErrorsHistogramConfig.Enabled = True
siStripFEDMonitor.AnyDAQProblemsHistogramConfig.Enabled = True
siStripFEDMonitor.AnyFEProblemsHistogramConfig.Enabled = True
siStripFEDMonitor.CorruptBuffersHistogramConfig.Enabled = True
siStripFEDMonitor.BadChannelStatusBitsHistogramConfig.Enabled = True
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
siStripFEDMonitor.FETimeDiffTIBHistogramConfig.Enabled = False
siStripFEDMonitor.FETimeDiffTOBHistogramConfig.Enabled = False
siStripFEDMonitor.FETimeDiffTECBHistogramConfig.Enabled = False
siStripFEDMonitor.FETimeDiffTECFHistogramConfig.Enabled = False
siStripFEDMonitor.ApveAddressHistogramConfig.Enabled = False
siStripFEDMonitor.FeMajAddressHistogramConfig.Enabled = False
#medians per APV for all channels, all events
siStripFEDMonitor.MedianAPV0HistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  NBins = cms.uint32(256),
  Min = cms.double(0),
  Max = cms.double(1024)
  )
siStripFEDMonitor.MedianAPV1HistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
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
  Enabled = cms.bool(False),
  #NBins = cms.uint32(101),
  #Min = cms.double(0),
  #Max = cms.double(101)
)
siStripFEDMonitor.nFEDsWithMissingFEsHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(101),
  #Min = cms.double(0),
  #Max = cms.double(101)
)
siStripFEDMonitor.nFEDsWithFEBadMajorityAddressesHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(101),
  #Min = cms.double(0),
  #Max = cms.double(101)
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
  Enabled = cms.bool(True),
  NBins = cms.uint32(250),
  Min = cms.double(0),
  Max = cms.double(500)
)
siStripFEDMonitor.nTotalBadActiveChannelsHistogramConfig = cms.PSet(
  Enabled = cms.bool(True),
  NBins = cms.uint32(250),
  Min = cms.double(0),
  Max = cms.double(500)
)
siStripFEDMonitor.nTotalBadChannelsvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(600),
  #Min = cms.double(0),
  #Max = cms.double(3600)
)
siStripFEDMonitor.nTotalBadActiveChannelsvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(600),
  #Min = cms.double(0),
  #Max = cms.double(3600)
)
siStripFEDMonitor.nFEDErrorsvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(600),
  #Min = cms.double(0),
  #Max = cms.double(3600)
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
  Enabled = cms.bool(False),
  #NBins = cms.uint32(600),
  #Min = cms.double(0),
  #Max = cms.double(3600)
)
siStripFEDMonitor.nAPVAddressErrorvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(600),
  #Min = cms.double(0),
  #Max = cms.double(3600)
)
siStripFEDMonitor.nUnlockedvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(600),
  #Min = cms.double(0),
  #Max = cms.double(3600)
)
siStripFEDMonitor.nOutOfSyncvsTimeHistogramConfig = cms.PSet(
  Enabled = cms.bool(False),
  #NBins = cms.uint32(600),
  #Min = cms.double(0),
  #Max = cms.double(3600)
)
