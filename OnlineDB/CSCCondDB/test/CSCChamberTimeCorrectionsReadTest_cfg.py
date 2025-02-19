import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCCHAMBERTIMECORRECTIONSREADTEST")
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.read = cms.EDAnalyzer("CSCChamberTimeCorrectionsReadTest")

process.p = cms.Path(process.read)


