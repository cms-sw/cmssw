import FWCore.ParameterSet.Config as cms

process = cms.Process("HEXDISPLAY")

process.load("CaloOnlineTools.EcalTools.ecalHexDisplay_cfi")

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    #untracked vstring fileNames = {'file:/data/scooper/data/grea07/40792E58-B757-DC11-8AB2-001617E30F46.root'}
    #fileNames = cms.untracked.vstring('file:/data/scooper/data/gren07/P5_Co.00029485.A.0.0.root')
    fileNames = cms.untracked.vstring('file:/data/scooper/data/cruzet3/7E738216-584D-DD11-9209-000423D6AF24.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(15)
)

process.counter = cms.OutputModule("AsciiOutputModule")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    )
)

process.p = cms.Path(process.hexDump)
process.ep = cms.EndPath(process.counter)

