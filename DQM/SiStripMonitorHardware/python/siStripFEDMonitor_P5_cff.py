import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorHardware.siStripFEDMonitor_cfi import *

#disable error output: enabled by default.
siStripFEDMonitor.PrintDebugMessages = 0
#lumi histogram
siStripFEDMonitor.ErrorFractionByLumiBlockHistogramConfig.Enabled = False
#Global/summary histograms
siStripFEDMonitor.FedEventSizeHistogramConfig.Enabled = True
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
siStripFEDMonitor.BadMajorityInPartitionHistogramConfig.Enabled = False
siStripFEDMonitor.FeMajFracTIBHistogramConfig.Enabled = False
siStripFEDMonitor.FeMajFracTOBHistogramConfig.Enabled = False
siStripFEDMonitor.FeMajFracTECBHistogramConfig.Enabled = False
siStripFEDMonitor.FeMajFracTECFHistogramConfig.Enabled = False
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
siStripFEDMonitor.BadMajorityAddressesDetailedHistogramConfig.Enabled = False
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
siStripFEDMonitor.MedianAPV0HistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(256),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(1024)
  )
siStripFEDMonitor.MedianAPV1HistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(256),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(1024)
  )      
#Error counting histograms
siStripFEDMonitor.nFEDErrorsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(101),
  Min = cms.untracked.double(-0.5),
  Max = cms.untracked.double(100.5)
)
siStripFEDMonitor.nFEDDAQProblemsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(101),
  Min = cms.untracked.double(-0.5),
  Max = cms.untracked.double(100.5)
)
siStripFEDMonitor.nFEDsWithFEProblemsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(101),
  Min = cms.untracked.double(-0.5),
  Max = cms.untracked.double(100.5)
)
siStripFEDMonitor.nFEDCorruptBuffersHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(101),
  Min = cms.untracked.double(-0.5),
  Max = cms.untracked.double(100.5)
)
#bins size number of FE Units/10, max is n channels
siStripFEDMonitor.nBadChannelStatusBitsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(250),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(500)
)
siStripFEDMonitor.nBadActiveChannelStatusBitsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(250),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(500)
)
siStripFEDMonitor.nFEDsWithFEOverflowsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(101),
  Min = cms.untracked.double(-0.5),
  Max = cms.untracked.double(100.5)
)
siStripFEDMonitor.nFEDsWithMissingFEsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(101),
  Min = cms.untracked.double(-0.5),
  Max = cms.untracked.double(100.5)
)
siStripFEDMonitor.nFEDsWithFEBadMajorityAddressesHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(101),
  Min = cms.untracked.double(-0.5),
  Max = cms.untracked.double(100.5)
)
siStripFEDMonitor.nUnconnectedChannelsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(250),
  Min = cms.untracked.double(6000),
  Max = cms.untracked.double(8000)
)
siStripFEDMonitor.nAPVStatusBitHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(250),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(500)
)
siStripFEDMonitor.nAPVErrorHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(250),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(500)
)
siStripFEDMonitor.nAPVAddressErrorHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(250),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(500)
)
siStripFEDMonitor.nUnlockedHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(250),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(500)
)
siStripFEDMonitor.nOutOfSyncHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(250),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(500)
)
siStripFEDMonitor.nTotalBadChannelsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(250),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(500)
)
siStripFEDMonitor.nTotalBadActiveChannelsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(False),
  #NBins = cms.untracked.uint32(250),
  #Min = cms.untracked.double(0),
  #Max = cms.untracked.double(500)
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
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(600),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(3600)
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
siStripFEDMonitor.FedMaxEventSizevsTimeHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(600),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(3600)
)
