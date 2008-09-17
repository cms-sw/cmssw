import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMOffline.EGamma.photonAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(


        '/store/relval/CMSSW_2_1_6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/024D606A-C078-DD11-BA5C-001D09F24498.root',
        '/store/relval/CMSSW_2_1_6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/02EDC451-C078-DD11-8C66-0019B9F707D8.root',
        '/store/relval/CMSSW_2_1_6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/06EEB605-C078-DD11-8FC3-001D09F27067.root',
        '/store/relval/CMSSW_2_1_6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/881B6B6A-C078-DD11-81E6-001D09F2514F.root',
        '/store/relval/CMSSW_2_1_6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/A2A5E967-EB78-DD11-8847-001617C3B6DE.root'

))



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('photonsMEtoEDMConverter.root')
)

#process.p1 = cms.Path(process.MEtoEDMConverter)
#process.p1 = cms.Path(process.photonAnalysis*process.MEtoEDMConverter)
process.p1 = cms.Path(process.photonAnalysis)
process.schedule = cms.Schedule(process.p1)

