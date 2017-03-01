import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'STARTHI53_LV1'
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_HIon', '')

process.GlobalTag.toGet.extend([
	cms.PSet(record = cms.string("HeavyIonRcd"),
		tag = cms.string("CentralityTable_HFtowers200_Glauber2010A_v5315x01_offline"),
		connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
		label = cms.untracked.string("HFtowers")
	),
])

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(100),
        )

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring('file:/tmp/azsigmon/hiReco_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO_101_1_dNN.root'),
	)

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")

process.load("RecoHI.HiCentralityAlgos.CentralityFilter_cfi")
process.centralityFilter.selectedBins = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)

process.output = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "test_filter.root" ),
    outputCommands = cms.untracked.vstring(
      'drop *',
      'keep recoCentrality_*_*_*',
      'keep *_centralityBin_*_*',
    ),
    SelectEvents = cms.untracked.PSet(
	SelectEvents = cms.vstring('p'),
    ),
)

process.p = cms.Path(process.centralityBin * process.centralityFilter)
process.out = cms.EndPath(process.output)
