import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMOffline.EGamma.photonAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(



    '/store/relval/CMSSW_3_1_0_pre1/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_30X_v1/0001/88C98CFC-06F8-DD11-AC35-000423D6CA6E.root'

##          '/store/relval/CMSSW_3_1_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/0CAA1EC0-F7F7-DD11-911D-000423D9970C.root',
##         '/store/relval/CMSSW_3_1_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/58AF1BC2-06F8-DD11-91E3-000423D985B0.root',
##         '/store/relval/CMSSW_3_1_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/DE195AF6-F8F7-DD11-A490-001617E30F56.root',
##          '/store/relval/CMSSW_3_1_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/E4EDF9CD-F7F7-DD11-825B-00304879FA4A.root'

##         '/store/relval/CMSSW_3_1_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/1270CC97-CEF7-DD11-906D-001617C3B79A.root',
##         '/store/relval/CMSSW_3_1_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/3E40D281-CEF7-DD11-9052-000423D944FC.root',
##         '/store/relval/CMSSW_3_1_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/AE8A1048-E2F7-DD11-8FC4-000423D98B08.root',
##         '/store/relval/CMSSW_3_1_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/B65320D4-06F8-DD11-B177-0016177CA7A0.root',
##         '/store/relval/CMSSW_3_1_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/E46D42A7-CEF7-DD11-BFC8-000423D99264.root'


))



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('photonsMEtoEDMConverter.root')
)

from DQMOffline.EGamma.photonAnalyzer_cfi import *
photonAnalysis.OutputMEsInRootFile = cms.bool(True)
photonAnalysis.OutputFileName = 'DQMPhotonsStandaloneForMC.root'
photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(True)
photonAnalysis.standAlone = cms.bool(True)



#process.p1 = cms.Path(process.MEtoEDMConverter)
#process.p1 = cms.Path(process.photonAnalysis*process.MEtoEDMConverter)
process.p1 = cms.Path(process.photonAnalysis)
process.schedule = cms.Schedule(process.p1)

