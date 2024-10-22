import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('TEST', eras.Run2_2018)

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    )
)

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring("/store/data/Run2018D/ZeroBias/RAW/v1/000/320/688/00000/601A721D-AD95-E811-B21A-FA163E28A50A.root"),
  #fileNames = cms.untracked.vstring("root://eoscms.cern.ch//eos/cms/store/group/phys_pps/sw_test_input/601A721D-AD95-E811-B21A-FA163E28A50A.root"),

  inputCommands = cms.untracked.vstring(
    'drop *',
    'keep FEDRawDataCollection_*_*_*'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1000)
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoPPS.Configuration.recoCTPPS_cff")

# define GT
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "106X_dataRun2_v26")

# override alignment settings
process.load("CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring(
  "RecoPPS/Local/test/re_alignment/align_base.xml",
  "RecoPPS/Local/test/re_alignment/align_corr.xml"
)

process.esPreferLocalAlignment = cms.ESPrefer("CTPPSRPAlignmentCorrectionsDataESSourceXML", "ctppsRPAlignmentCorrectionsDataESSourceXML")

# track plotter
process.ctppsTrackDistributionPlotter = cms.EDAnalyzer("CTPPSTrackDistributionPlotter",
  tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
  outputFile = cms.string("output_tracks_full.root")
)

# processing sequences
process.path = cms.Path(
  process.ctppsRawToDigi
  * process.recoCTPPS
  * process.ctppsTrackDistributionPlotter
)
