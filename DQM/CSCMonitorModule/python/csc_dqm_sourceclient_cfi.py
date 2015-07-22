import FWCore.ParameterSet.Config as cms

#--------------------------
# DQM Module
#--------------------------

from DQM.CSCMonitorModule.csc_dqm_masked_hw_cfi import *

dqmCSCClient = cms.EDAnalyzer("CSCMonitorModule",

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
      '-/All_Readout_Errors/',
      '-/FEDBufferSize/',
      '-/FEDEntries/',
      '-/FEDFatal/',
      '-/FED_Stats/',
      '-/FEDNonFatal/',
      '-/FEDFormatFatal/',
      '-/FEDFormat_Errors/',
      '-/FED_DDU_L1A_.*$/', 
      '-/^FED_[0-9]+/',
      '-/^DMB_.*$/',
      '+/DDU_[0-9]+/',
      '-/CSC_[0-9]+_[0-9]+/',
      '+/CSC_[0-9]+_[0-9]+\/BinCheck_ErrorStat_Table/',
      '+/CSC_[0-9]+_[0-9]+\/BinCheck_DataFlow_Problems_Table/',
      '+/CSC_[0-9]+_[0-9]+\/ALCT[01]_dTime/',
      '+/CSC_[0-9]+_[0-9]+\/ALCT[01]_Quality/',
      '+/CSC_[0-9]+_[0-9]+\/ALCT[01]_Pattern_Distr/',
      '+/CSC_[0-9]+_[0-9]+\/AFEB_RawHits_TimeBins/',
      '+/CSC_[0-9]+_[0-9]+\/ALCT_Number_Of_Layers_With_Hits/',
      '+/CSC_[0-9]+_[0-9]+\/CFEB_SCA_CellPeak_Time/',
      '+/CSC_[0-9]+_[0-9]+\/CFEB_SCA_Cell_Peak_Ly_[0-9]/',
      '+/CSC_[0-9]+_[0-9]+\/CFEB_Comparators_TimeSamples/',
      '+/CSC_[0-9]+_[0-9]+\/ALCT_Match_Time/',
      '+/CSC_[0-9]+_[0-9]+\/CLCT[0-9]_dTime/',
      '+/CSC_[0-9]+_[0-9]+\/CLCT_Number_Of_Layers_With_Hits/',
      '+/CSC_[0-9]+_[0-9]+\/CLCT[0-9]_Half_Strip_Quality_Distr/',
      '+/CSC_[0-9]+_[0-9]+\/Chamber_Event_Display_No[1]/'
    )
  )

)


