import FWCore.ParameterSet.Config as cms

process = cms.Process("RecoSTAMuon")
process.load("RecoMuon.Configuration.RecoMuon_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_V9::All'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_10/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/A8EB0765-BC9A-DD11-AB0B-001A92971B64.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('cout')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/RecoSTAMuons.root')
)

## Analyzer to produce pT and 1/pT resolution plots
#process.STAMuonAnalyzer = cms.EDAnalyzer("STAMuonAnalyzer",
#                                         DataType = cms.untracked.string('SimData'),
#                                         StandAloneTrackCollectionLabel = cms.untracked.string('standAloneMuons'),
#                                         MuonSeedCollectionLabel = cms.untracked.string('MuonSeed'),
#                                         rootFileName = cms.untracked.string('STAMuonAnalyzer.root')
#                                         )

process.p = cms.Path(process.MuonSeed * process.standAloneMuons)                             ## default path (no analyzer)
#process.p = cms.Path(process.MuonSeed * process.standAloneMuons * process.STAMuonAnalyzer)  ## path with analyzer
process.this_is_the_end = cms.EndPath(process.out)
