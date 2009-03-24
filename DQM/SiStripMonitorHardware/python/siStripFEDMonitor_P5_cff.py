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
siStripFEDMonitor.nBadActiveChannelStatusBitsHistogramConfig = cms.untracked.PSet(
  Enabled = cms.untracked.bool(True),
  NBins = cms.untracked.uint32(353),
  Min = cms.untracked.double(0),
  Max = cms.untracked.double(422401)
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
