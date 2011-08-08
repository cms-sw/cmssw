import FWCore.ParameterSet.Config as cms;

laserSorter = cms.EDAnalyzer("LaserSorter",
  outputDir  = cms.string("out"),
  fedSubDirs = cms.vstring(
    "Unknown",
    "EE-7",  "EE-8",  "EE-9",  "EE-1",  "EE-2",
    "EE-3",  "EE-4",  "EE-5",  "EE-6",  "EB-1",
    "EB-2",  "EB-3",  "EB-4",  "EB-5",  "EB-6",
    "EB-7",  "EB-8",  "EB-9",  "EB-10", "EB-11",
    "EB-12", "EB-13", "EB-14", "EB-15", "EB-16",
    "EB-17", "EB-18", "EB+1",  "EB+2",  "EB+3",
    "EB+4",  "EB+5",  "EB+6",  "EB+7",  "EB+8",
    "EB+9",  "EB+10", "EB+11", "EB+12", "EB+13",
    "EB+14", "EB+15", "EB+16", "EB+17", "EB+18",
    "EE+7",  "EE+8",  "EE+9",  "EE+1",  "EE+2",
    "EE+3",  "EE+4",  "EE+5", "EE+6"),
  timeLogFile = cms.untracked.string("laserSortingTime.txt"),
  disableOutput = cms.untracked.bool(False),
  outputListFile = cms.untracked.string("lmfFileList.txt"),
  verbosity = cms.untracked.int32(0),
  #limit on "no fully readout dcc error" messages per run
  maxFullReadoutDccError = cms.int32(10),
  #limit on "No ECAL DCC Data" messages per run
  maxNoEcalDataMess = cms.int32(10)
)

