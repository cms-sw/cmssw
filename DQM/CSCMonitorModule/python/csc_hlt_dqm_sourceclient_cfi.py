import FWCore.ParameterSet.Config as cms

#--------------------------
# DQM Module
#--------------------------

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
cscDQMEvF = DQMEDAnalyzer('CSCMonitorModule',

  BOOKING_XML_FILE = cms.FileInPath('DQM/CSCMonitorModule/data/emuDQMBooking.xml'),
  InputObjects = cms.untracked.InputTag("rawDataCollector"),

  EventProcessor = cms.untracked.PSet(
    PROCESS_DDU = cms.untracked.bool(False),
    PROCESS_CSC = cms.untracked.bool(False),
    PROCESS_EFF_HISTOS = cms.untracked.bool(False),
    PROCESS_EFF_PARAMETERS = cms.untracked.bool(False),
    BINCHECKER_CRC_ALCT = cms.untracked.bool(True),
    BINCHECKER_CRC_CLCT = cms.untracked.bool(True),
    BINCHECKER_CRC_CFEB = cms.untracked.bool(True),
    BINCHECKER_MODE_DDU = cms.untracked.bool(False),
    BINCHECKER_OUTPUT   = cms.untracked.bool(False),
    FRAEFF_AUTO_UPDATE  = cms.untracked.bool(False),
    FRAEFF_SEPARATE_THREAD  = cms.untracked.bool(False),
    FOLDER_EMU = cms.untracked.string('CSC/FEDIntegrity/'),
    FOLDER_FED = cms.untracked.string(''),
    FOLDER_DDU = cms.untracked.string(''),
    FOLDER_CSC = cms.untracked.string(''),
    FOLDER_PAR = cms.untracked.string(''),
    DDU_CHECK_MASK = cms.untracked.uint32(0xFFFFDFFF),
    DDU_BINCHECK_MASK = cms.untracked.uint32(0x16EBF7F6),
    BINCHECK_MASK = cms.untracked.uint32(0x16EBF7F6),
    FRAEFF_AUTO_UPDATE_START = cms.untracked.uint32(5),
    FRAEFF_AUTO_UPDATE_FREQ = cms.untracked.uint32(200),
    EFF_COLD_THRESHOLD = cms.untracked.double(0.1),
    EFF_COLD_SIGFAIL = cms.untracked.double(5.0),
    EFF_HOT_THRESHOLD = cms.untracked.double(0.1),
    EFF_HOT_SIGFAIL = cms.untracked.double(5.0),
    EFF_ERR_THRESHOLD = cms.untracked.double(0.1),
    EFF_ERR_SIGFAIL = cms.untracked.double(5.0),
    EFF_NODATA_THRESHOLD = cms.untracked.double(0.1),
    EFF_NODATA_SIGFAIL = cms.untracked.double(5.0),
    EVENTS_ECHO = cms.untracked.uint32(1000),
    MO_FILTER = cms.untracked.vstring(
      '-/^.*$/',
      '+/FEDEntries/',
      '+/FEDFatal/',
      '+/FEDFormatFatal/',
      '+/FEDNonFatal/',
      '+/FEDBufferSize/',
      '+/FEDTotalEventSize/',
      '+/FEDFormat_Errors/',
      '+/^CSC_Reporting$/',
      '+/^CSC_Format_Errors$/',
      '+/^CSC_Format_Warnings$/',
      '+/^CSC_L1A_out_of_sync$/',
      '+/^CSC_wo_ALCT$/',
      '+/^CSC_wo_CFEB$/',
      '+/^CSC_wo_CLCT$/'
    )
  )

)

