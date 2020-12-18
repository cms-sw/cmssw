import FWCore.ParameterSet.Config as cms
process = cms.Process("GeometryInfo")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

# load alignment corrections
process.load("CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring("CondFormats/PPSObjects/xml/sample_alignment_corrections.xml")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.MeasuredFiles = cms.vstring("CondFormats/PPSObjects/xml/sample_alignment_corrections.xml")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = cms.vstring("CondFormats/PPSObjects/xml/sample_alignment_corrections.xml")

# no events to process
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.ctppsAlignmentInfo = cms.EDAnalyzer("CTPPSAlignmentInfo",
    alignmentType = cms.untracked.string("real"),
)

process.p = cms.Path(
    process.ctppsAlignmentInfo
)
