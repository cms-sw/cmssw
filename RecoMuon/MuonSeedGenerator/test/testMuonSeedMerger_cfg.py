import FWCore.ParameterSet.Config as cms

process = cms.Process("TestMuonSeedMerger")

# Messages
process.load("RecoMuon.Configuration.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cff")


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0000/7E8CEFD2-EE5F-DD11-B131-000423D6CA6E.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('TESTMergedSeed.root')
                               )

process.p = cms.Path(process.standAloneMuonSeeds)
process.this_is_the_end = cms.EndPath(process.out)
process.GlobalTag.globaltag = 'IDEAL_V5::All'
