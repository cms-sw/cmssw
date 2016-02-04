import FWCore.ParameterSet.Config as cms

process = cms.Process('testRecoEcal')
process.load('RecoEcal.Configuration.RecoEcal_cff')
process.load("Configuration.StandardSequences.Geometry_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(1),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
#        '/store/mc/Summer08/Zee_M20/GEN-SIM-RECO/IDEAL_V9_reco-v3/0001/0246B2F4-6FF3-DD11-94D3-0030487F92B5.root'
        '/store/relval/CMSSW_3_1_0_pre4/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_30X_v1/0003/F602BF32-2916-DE11-B4DE-000423D8FA38.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_30X_v1/0003/F698554C-AB16-DE11-8C19-001617E30D06.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_*_*_testRecoEcal'),
    fileName = cms.untracked.string('output_testRecoEcal.root')
)

process.p = cms.Path(process.ecalClusters)
process.outpath = cms.EndPath(process.out)

