import FWCore.ParameterSet.Config as cms

#--------------------------
# DQM Module
#--------------------------

from DQM.CSCMonitorModule.csc_dqm_masked_hw_cfi import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmCSCClient = DQMEDAnalyzer('CSCMonitorModule',

  BOOKING_XML_FILE = cms.FileInPath('DQM/CSCMonitorModule/data/emuDQMBooking.xml'),
  InputObjects = cms.untracked.InputTag("rawDataCollector"),
  PREBOOK_EFF_PARAMS = cms.untracked.bool(True),
  MASKEDHW = CSCMaskedHW,

  EventProcessor = cms.untracked.PSet(
    PROCESS_DDU = cms.untracked.bool(True),
    PROCESS_CSC = cms.untracked.bool(True),
    PROCESS_EFF_HISTOS = cms.untracked.bool(True),
    PROCESS_EFF_PARAMETERS = cms.untracked.bool(True),
    BINCHECKER_CRC_ALCT = cms.untracked.bool(True),
    BINCHECKER_CRC_CLCT = cms.untracked.bool(True),
    BINCHECKER_CRC_CFEB = cms.untracked.bool(True),
    BINCHECKER_MODE_DDU = cms.untracked.bool(False),
    BINCHECKER_OUTPUT   = cms.untracked.bool(False),
    FRAEFF_AUTO_UPDATE  = cms.untracked.bool(True),
    FRAEFF_SEPARATE_THREAD  = cms.untracked.bool(False),
    FOLDER_EMU = cms.untracked.string('CSC/Summary/'),
    FOLDER_FED = cms.untracked.string('CSC/FED/'),
    FOLDER_DDU = cms.untracked.string('CSC/DDU/'),
    FOLDER_CSC = cms.untracked.string('CSC/CSC/'),
    FOLDER_PAR = cms.untracked.string('CSC/EventInfo/reportSummaryContents/'),
    DDU_CHECK_MASK = cms.untracked.uint32(0xFFFFDFFF),
    DDU_BINCHECK_MASK = cms.untracked.uint32(0x16EBF7F6),
    BINCHECK_MASK = cms.untracked.uint32(0x16EBF7F6),
    FRAEFF_AUTO_UPDATE_START = cms.untracked.uint32(5),
    FRAEFF_AUTO_UPDATE_FREQ = cms.untracked.uint32(200),
    EFF_COLD_THRESHOLD = cms.untracked.double(0.1),
    EFF_COLD_SIGFAIL = cms.untracked.double(1.5),
    EFF_HOT_THRESHOLD = cms.untracked.double(2.0),
    EFF_HOT_SIGFAIL = cms.untracked.double(10.0),
    EFF_ERR_THRESHOLD = cms.untracked.double(0.1),
    EFF_ERR_SIGFAIL = cms.untracked.double(5.0),
    EFF_NODATA_THRESHOLD = cms.untracked.double(0.5),
    EFF_NODATA_SIGFAIL = cms.untracked.double(10.0),
    EVENTS_ECHO = cms.untracked.uint32(1000),
    MO_FILTER = cms.untracked.vstring(
      '+/^.*$/',
    )
  )

)


