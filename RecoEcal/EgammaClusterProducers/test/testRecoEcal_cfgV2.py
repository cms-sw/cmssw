import FWCore.ParameterSet.Config as cms

process = cms.Process('testRecoEcal')
process.load('RecoEcal.Configuration.RecoEcal_cff')
process.load("Configuration.StandardSequences.Geometry_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
## global tags:
#process.GlobalTag.globaltag = cms.string('GR_R_37X_V6A::All')
process.GlobalTag.globaltag = cms.string('START38_V8::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


process.source = cms.Source("PoolSource",
#    debugVerbosity = cms.untracked.uint32(1),
#    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_9_0_pre1/RelValZEE/GEN-SIM-RECO/START38_V8-v1/0009/1A21AD3E-F89A-DF11-A955-002618943865.root',
#    '/store/data/Run2010A/EG/RECO/Jun14thReReco_v1/0000/062F305D-A278-DF11-A9B3-003048F0E3B2.root',
    #'/store/relval/CMSSW_3_1_0_pre4/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_30X_v1/0003/F602BF32-2916-DE11-B4DE-000423D8FA38.root',
    #'/store/relval/CMSSW_3_1_0_pre4/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_30X_v1/0003/F698554C-AB16-DE11-8C19-001617E30D06.root'
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

