import FWCore.ParameterSet.Config as cms

# set process name and set conditions for year 2018
from Configuration.StandardSequences.Eras import eras
process = cms.Process('TEST', eras.Run2_2018)

# chose global tag GT = list of DB payloads containing conditions data
from Configuration.AlCa.GlobalTag import GlobalTag
from CondCore.CondDB.CondDB_cfi import *
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag = GlobalTag(process.GlobalTag, "112X_dataRun2_v6")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring("cout"),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string("WARNING")
  )
)

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring("root://eoscms.cern.ch//eos/cms/store/group/phys_pps/sw_test_input/3204EE5B-C298-E611-BC39-02163E01448F.root"),

  inputCommands = cms.untracked.vstring(
    'drop *',
    'keep FEDRawDataCollection_*_*_*'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1) # by setting a positive number here, you can limit the number of events to be processed
)

# load default alignment settings
process.load("CalibPPS.ESProducers.ctppsAlignment_cff")

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoPPS.Configuration.recoCTPPS_cff")

# reconstruction plotter
process.ctppsProtonReconstructionPlotter = cms.EDAnalyzer("CTPPSProtonReconstructionPlotter",
  tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
  tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons", "singleRP"),
  tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons", "multiRP"),

  rpId_45_F = cms.uint32(3),
  rpId_45_N = cms.uint32(2),
  rpId_56_N = cms.uint32(102),
  rpId_56_F = cms.uint32(103),

  outputFile = cms.string("reco_plots.root")
)

# processing sequences
process.path = cms.Path(
  process.ctppsRawToDigi
  * process.recoCTPPS
  * process.ctppsProtonReconstructionPlotter
)
