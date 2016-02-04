import FWCore.ParameterSet.Config as cms

from DQM.CSCMonitorModule.csc_dqm_masked_hw_cfi import *

dqmCSCOfflineClient = cms.EDAnalyzer("CSCOfflineClient",

  MASKEDHW = CSCMaskedHW,

  EventProcessor = cms.untracked.PSet(
    PROCESS_DDU = cms.untracked.bool(True),
    PROCESS_CSC = cms.untracked.bool(True),
    PROCESS_EFF_HISTOS = cms.untracked.bool(True),
    IN_FULL_STANDBY = cms.untracked.bool(False),
    PROCESS_EFF_PARAMETERS = cms.untracked.bool(True),
    BINCHECKER_CRC_ALCT = cms.untracked.bool(True),
    BINCHECKER_CRC_CLCT = cms.untracked.bool(True),
    BINCHECKER_CRC_CFEB = cms.untracked.bool(True),
    BINCHECKER_MODE_DDU = cms.untracked.bool(False),
    BINCHECKER_OUTPUT   = cms.untracked.bool(False),
    FRAEFF_AUTO_UPDATE  = cms.untracked.bool(True),
    FRAEFF_SEPARATE_THREAD  = cms.untracked.bool(False),
    FOLDER_EMU = cms.untracked.string('CSC/Summary/'),
    FOLDER_DDU = cms.untracked.string('CSC/DDU/'),
    FOLDER_CSC = cms.untracked.string('CSC/CSC/'),
    FOLDER_PAR = cms.untracked.string('CSC/EventInfo/reportSummaryContents/'),
    DDU_CHECK_MASK = cms.untracked.uint32(0xFFFFDFFF),
    DDU_BINCHECK_MASK = cms.untracked.uint32(0x16EBF7F6),
    BINCHECK_MASK = cms.untracked.uint32(0x16EBF7F6),
    FRAEFF_AUTO_UPDATE_START = cms.untracked.uint32(5),
    FRAEFF_AUTO_UPDATE_FREQ = cms.untracked.uint32(200),
    EFF_COLD_THRESHOLD = cms.untracked.double(0.1),
    EFF_COLD_SIGFAIL = cms.untracked.double(2.0),
    EFF_HOT_THRESHOLD = cms.untracked.double(2.0),
    EFF_HOT_SIGFAIL = cms.untracked.double(5.0),
    EFF_ERR_THRESHOLD = cms.untracked.double(0.1),
    EFF_ERR_SIGFAIL = cms.untracked.double(5.0),
    EFF_NODATA_THRESHOLD = cms.untracked.double(0.99),
    EFF_NODATA_SIGFAIL = cms.untracked.double(5.0),
    EVENTS_ECHO = cms.untracked.uint32(1000),
    MO_FILTER = cms.untracked.vstring(
      '-/^.*$/'
    )
  )

)


