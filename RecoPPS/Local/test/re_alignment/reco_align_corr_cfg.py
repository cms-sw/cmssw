import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('ReAlignment', eras.Run2_2018)

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
  fileNames = cms.untracked.vstring("file://output_base.root"),
)

# load alignment correction
process.load("CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring(
  "RecoPPS/Local/test/re_alignment/align_corr.xml"
)

process.esPreferLocalAlignment = cms.ESPrefer("CTPPSRPAlignmentCorrectionsDataESSourceXML", "ctppsRPAlignmentCorrectionsDataESSourceXML")

# track re-alignment module
process.load("RecoPPS.Local.ppsLocalTrackLiteReAligner_cfi")

# track plotter
process.ctppsTrackDistributionPlotter = cms.EDAnalyzer("CTPPSTrackDistributionPlotter",
  tagTracks = cms.InputTag("ppsLocalTrackLiteReAligner"),
  outputFile = cms.string("output_tracks_corr.root")
)

# processing sequences
process.path = cms.Path(
  process.ppsLocalTrackLiteReAligner
  * process.ctppsTrackDistributionPlotter
)
