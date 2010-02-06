import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("DQMOffline.EGamma.electronAnalyzerSequence_cff")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/2/RelVal-RelValSingleElectronPt35-1212355159-IDEAL_V1-2nd/0000/02FE32F3-A630-DD11-868D-001617E30F4C.root', 
        '/store/relval/2008/6/2/RelVal-RelValSingleElectronPt35-1212355159-IDEAL_V1-2nd/0000/9CD77E8E-A630-DD11-9C8B-000423D98DD4.root', 
        '/store/relval/2008/6/2/RelVal-RelValSingleElectronPt35-1212355159-IDEAL_V1-2nd/0000/D071B9F8-A630-DD11-A840-000423D6CA6E.root')
)

process.DQMStore = cms.Service("DQMStore")

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('MEtoEDMConverter.root')
)

process.p1 = cms.Path(process.electronAnalyzerSequence*process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p1,process.outpath)

process.PoolSource.fileNames = ['/store/relval/CMSSW_2_1_2/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_V6_v3/0001/0C345214-B56A-DD11-9B49-000423D999CA.root', '/store/relval/CMSSW_2_1_2/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_V6_v3/0001/C20CE1D5-8F6A-DD11-A99E-001617E30F58.root']


