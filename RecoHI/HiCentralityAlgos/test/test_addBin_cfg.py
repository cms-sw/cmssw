import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'STARTHI53_LV1'
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_hi', '')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(100),
        )

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring('file:/tmp/azsigmon/hiReco_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO_101_1_dNN.root'),
	)

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")

process.output = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "test.root" ),
    outputCommands = cms.untracked.vstring(
      'drop *',
      'keep recoCentrality_*_*_*',
      'keep *_centralityBin_*_*',
    )
)

process.p = cms.Path(process.centralityBin)
process.out = cms.EndPath(process.output)
