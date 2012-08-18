##-- Starting
import FWCore.ParameterSet.Config as cms
process = cms.Process("DQM")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.EventContent.EventContent_cff')


##-- DQM Loading
# DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
# DQM Sources
process.load("CondCore.DBCommon.CondDBSetup_cfi")
# DQMOffline/Trigger
process.load("DQMOffline.Trigger.FourVectorHLTOffline_cfi")
process.load("DQMOffline.Trigger.FourVectorHLTOfflineClient_cfi")
#process.load("DQMOffline.Trigger.JetMETHLTOfflineSource_cfi")
process.load("DQMOffline.Trigger.JetMETHLTOfflineAnalyzer_cff")
process.load("DQMOffline.Trigger.JetMETHLTOfflineClient_cfi")
process.load("DQMOffline.Trigger.HLTJetMETQualityTester_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
#
process.DQMStore.verbose = 0 #0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online' #Online
process.dqmSaver.saveByRun = 1 #0
process.dqmSaver.saveAtJobEnd = True


##-- GlobalTag
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR_R_52_V7::All'
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = cms.string( autoCond[ 'com10' ] )


##-- L1
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.L1T1coll=process.hltLevel1GTSeed.clone()
process.L1T1coll.L1TechTriggerSeeding = cms.bool(True)
process.L1T1coll.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')
#process.L1T1coll.L1SeedsLogicalExpression = cms.string(' (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')


##-- Filters
process.noscraping = cms.EDFilter(
    "FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False),
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.25)
)
process.primaryVertexFilter = cms.EDFilter(
    "GoodVertexFilter",
    vertexCollection = cms.InputTag('offlinePrimaryVertices'),
    minimumNDOF = cms.uint32(4) ,
    maxAbsZ = cms.double(24),
    maxd0 = cms.double(2)
)


##-- DQMOffline/Trigger
from DQMOffline.Trigger.FourVectorHLTOfflineClient_cfi import *
from DQMOffline.Trigger.JetMETHLTOfflineClient_cfi import *
process.load("DQMServices.Components.DQMStoreStats_cfi")
process.hltclient = cms.Sequence(hltFourVectorClient)
hltFourVectorClient.prescaleLS = cms.untracked.int32(-1)
hltFourVectorClient.monitorDir = cms.untracked.string('')
hltFourVectorClient.prescaleEvt = cms.untracked.int32(1)


##-- Source
# Note: We need RECO here (not AOD), because of JetHelper Class
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)                             
)
process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring())
dataset = "MET"
if (dataset.find("MET")==0):
    process.source.fileNames.extend([
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/D4C2D7A1-59A3-E111-8E62-003048D37694.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/CACD5368-57A3-E111-B292-003048F024DE.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/801A69EF-53A3-E111-BAC9-003048D2BB58.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/66D6CCE9-53A3-E111-87CD-002481E0D790.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/5A76C4CC-58A3-E111-AFED-001D09F27067.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/2E977848-52A3-E111-9EBF-BCAEC5329709.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/2E7E2B8C-51A3-E111-B818-001D09F28EA3.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/26038F94-54A3-E111-97A9-003048D2C0F0.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/0ED82FA8-51A3-E111-A534-003048F11114.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/00124554-5AA3-E111-9004-001D09F2AD84.root',
        ])
elif (dataset.find("JetMon")==0):
    process.source.fileNames.extend([
        '/store/data/Run2012B/JetMon/RECO/PromptReco-v1/000/194/429/E6B6E24E-5AA3-E111-8828-003048F118AA.root',
        '/store/data/Run2012B/JetMon/RECO/PromptReco-v1/000/194/429/E48B31FC-62A3-E111-B3BE-003048D375AA.root',
        '/store/data/Run2012B/JetMon/RECO/PromptReco-v1/000/194/429/D209A34F-5AA3-E111-93FF-003048F118C6.root',
        '/store/data/Run2012B/JetMon/RECO/PromptReco-v1/000/194/429/C2D8ED6A-5DA3-E111-9174-003048F118C6.root',
        '/store/data/Run2012B/JetMon/RECO/PromptReco-v1/000/194/429/A2E73857-5FA3-E111-87FC-003048F1182E.root',
        '/store/data/Run2012B/JetMon/RECO/PromptReco-v1/000/194/429/741F9531-60A3-E111-991D-BCAEC518FF54.root',
        '/store/data/Run2012B/JetMon/RECO/PromptReco-v1/000/194/429/6A240F4F-69A3-E111-9068-5404A640A648.root',
        '/store/data/Run2012B/JetMon/RECO/PromptReco-v1/000/194/429/4EB20BA8-56A3-E111-A8A3-001D09F25479.root',
        '/store/data/Run2012B/JetMon/RECO/PromptReco-v1/000/194/429/28909B85-5EA3-E111-817B-001D09F24399.root',
        '/store/data/Run2012B/JetMon/RECO/PromptReco-v1/000/194/429/1CC2654E-69A3-E111-B303-001D09F28E80.root',
        ])
elif (dataset.find("JetHT")==0):
    process.source.fileNames.extend([
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/F82EC951-69A3-E111-AD74-001D09F253D4.root',
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/F06F8D22-6EA3-E111-AD78-BCAEC5329701.root',
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/CEA0198D-5EA3-E111-9F80-003048D2BC52.root',
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/CC22C24F-69A3-E111-B8A7-0019B9F72F97.root',
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/C4BBE1EA-63A3-E111-830E-001D09F25479.root',
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/AA12FEB5-76A3-E111-8994-BCAEC532971F.root',
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/A62ED1B8-60A3-E111-A202-0019B9F4A1D7.root',
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/9C485185-68A3-E111-BE90-003048F024FE.root',
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/80E4A152-6BA3-E111-A42D-0025901D5DB8.root',
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/4A1A414F-70A3-E111-ABF0-485B39897227.root',
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/3E537337-60A3-E111-823F-5404A63886EB.root',
        '/store/data/Run2012B/JetHT/RECO/PromptReco-v1/000/194/429/2CB8EC6D-5DA3-E111-9229-E0CB4E4408C4.root',
        ])
elif (dataset.find("reference")==0):
    process.source.fileNames.extend([
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/0A8D0A87-C7D1-E111-BB91-001A92810AC0.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/02A5AF56-CAD1-E111-B032-0026189438B1.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/046A20C7-C7D1-E111-88C2-0026189438C9.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/0E8CD563-C7D1-E111-BB63-002618943811.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/087C0DB2-CFD1-E111-B67F-0018F3C3E3A6.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/0ADBCF73-C7D1-E111-8DEE-002618943966.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/1E1443E9-C8D1-E111-8422-00261894393E.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/1E96A911-C7D1-E111-B0EC-003048678AFA.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2655603F-CBD1-E111-B505-001A928116C4.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/28CEE6D8-C1D1-E111-B750-001A92971B26.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/0AF4B820-C8D1-E111-82AA-002618943886.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/0CCFAD86-C7D1-E111-AE30-001A92810ACA.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/18290821-C7D1-E111-A0E6-001A92811726.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/1AD366BD-CAD1-E111-8AF9-001A92811736.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2E3E34CB-CAD1-E111-9D1B-0026189438C1.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/323C458A-CBD1-E111-9FE8-001BFCDBD154.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/36D2123B-C7D1-E111-AFE0-003048678B44.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2A282D37-CBD1-E111-A37B-00304867915A.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2A9E0754-C7D1-E111-975B-002618943863.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/40E61297-CAD1-E111-8F9D-001A92971B90.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2AB03D70-C7D1-E111-A3D9-003048FFD740.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2C5D3008-CAD1-E111-859B-00261894397E.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/3CB31D71-C2D1-E111-A77B-001A928116BC.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/40AF624C-CAD1-E111-A1F2-00261894385D.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/42668A88-CBD1-E111-BD40-002618943926.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/5249CA2C-C7D1-E111-B978-001A92810AA2.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/42C2FA7F-CBD1-E111-B6CC-001A92971B38.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/44410558-C7D1-E111-A6E8-003048679046.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/62F79061-CCD1-E111-8800-001731EF61B4.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/66AE7A0D-C7D1-E111-AA6D-002618943800.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/66D46034-CBD1-E111-BCE2-0026189438D5.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/48C98DE9-C6D1-E111-B80A-0026189438E6.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/4CDC4AE6-C8D1-E111-9602-002618943858.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/52F08646-C7D1-E111-B2A6-001BFCDBD11E.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/781C0F71-C7D1-E111-98B4-001A928116BC.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/5A05D29E-CAD1-E111-B5EF-003048678FFA.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/8247BF73-CBD1-E111-8AFA-0018F3D0960C.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/82AF42BF-CBD1-E111-8F49-001A92971B5E.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/6A976463-C7D1-E111-81E7-0018F3D096D8.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/6C21EDF9-C6D1-E111-A2E7-003048678BEA.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/9406BBA2-CAD1-E111-A22A-001A92971AAA.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/769148D9-CBD1-E111-8D4C-001A92811736.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/7E7D29E9-C8D1-E111-8BAB-002618943915.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/9C0BA086-C7D1-E111-81E2-001A92810A9E.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/A671B9D6-CAD1-E111-B216-001A92971B90.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/84F49DBB-C8D1-E111-B3DC-00261894397D.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/8AE1E172-CBD1-E111-8353-003048679248.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/A89BCD54-CBD1-E111-85AC-003048FFD71E.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/96FEF6FC-C6D1-E111-9DB5-001A928116CE.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/AC808A90-CBD1-E111-AEC3-0026189438C4.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/B4369931-C7D1-E111-A930-0026189438C9.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/9A7B5CB8-CAD1-E111-9FB5-001A92811742.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/A820436D-C2D1-E111-B0A1-0018F3D09702.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/A8776D82-C7D1-E111-8388-00304867BFB2.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/AADF71CF-CAD1-E111-8CD6-0018F3D0968C.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/BECC10DE-CBD1-E111-9B0A-001BFCDBD154.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/C00F937A-C7D1-E111-A511-003048678B7C.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/B6E831EA-C1D1-E111-8091-0018F3D0968C.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/BA0C8AEA-C8D1-E111-8FA5-003048D3C010.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/CC6D8823-C9D1-E111-B316-001A92971B94.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/CC88DDD2-C2D1-E111-9AF5-001A92971B08.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/D0161136-C7D1-E111-9B76-001A92971B12.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/BE38F451-C7D1-E111-86FC-003048678FAE.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/D49D585E-C7D1-E111-8C78-0026189438DD.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/BE94A0D5-C7D1-E111-99E3-0026189438E6.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/DEC7EF75-CBD1-E111-B2CE-0018F3D09702.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/C0D1758D-C9D1-E111-AB65-001A928116C0.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/EE0F62F5-CCD1-E111-8C2A-00304867D446.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/EE974FAF-CAD1-E111-81D9-0018F3C3E3A6.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/F082E2A4-CDD1-E111-A19C-003048D3C010.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/F20CCD7F-C7D1-E111-B0F7-003048678FFA.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/C4BFFA3B-C7D1-E111-9182-001A92810A98.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/F8443B6A-C7D1-E111-B2D8-0018F3D09702.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/D20127ED-C1D1-E111-9EB7-0018F3D09636.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/D83B3869-C7D1-E111-BBED-001BFCDBD1BA.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/EC9F9E55-CAD1-E111-8686-00261894396B.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/F2D4FC78-C7D1-E111-9562-0018F3D096BC.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/FCA4CF95-CBD1-E111-8074-00304867D446.root',
        '/store/relval/CMSSW_5_2_6-reference_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/FE1130C5-CCD1-E111-B84E-0018F3C3E3A6.root'
        ])
elif (dataset.find("newcondition")==0):
    process.source.fileNames.extend([
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/008D0C53-C7D1-E111-9373-0026189438CB.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/00A0B0B1-CAD1-E111-95F2-002618943863.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/082F7655-CAD1-E111-8B50-003048678B44.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/02AA8C7E-C5D1-E111-A6E8-001A92811742.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/046E9164-C9D1-E111-9665-0018F3D09634.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/065CBBF4-C9D1-E111-8331-001A92810AEC.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/06F24B41-C5D1-E111-8822-00261894397E.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/0A61C045-C5D1-E111-B6C8-0026189438C4.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2465FBB9-CAD1-E111-91B6-0026189437FA.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/0ACC745E-C9D1-E111-92EB-003048678D86.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2AA69E7B-C9D1-E111-9DBE-00261894397D.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/14E7C560-C9D1-E111-BE66-00261894385D.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/18DB0D69-C7D1-E111-8B1B-00261894388A.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2CEBD57F-C9D1-E111-82EF-0018F3D096E0.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/30940F09-C8D1-E111-81A8-0018F3D09636.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/18E84C50-CBD1-E111-BE49-002618943838.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/32366A12-CAD1-E111-8500-001A928116D0.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2808FC3B-C9D1-E111-B210-001A92811726.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2C5F6095-C9D1-E111-9E25-0026189438FE.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/2C5FAAA9-CAD1-E111-95BA-001A9281173A.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/32264425-CAD1-E111-8B0E-002618943966.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/346C823A-C5D1-E111-A667-003048679162.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/5279E9CB-C9D1-E111-8C1D-003048FFD732.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/409CA34B-C5D1-E111-9BF7-003048678B34.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/487C45F9-C4D1-E111-A2DD-0026189438FE.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/4AE54EAE-C8D1-E111-983D-003048679180.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/5A4BD344-C9D1-E111-B0E6-001A928116EA.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/5C9C264D-CAD1-E111-AE7E-003048679248.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/4C6346BF-C7D1-E111-9089-003048FFD71A.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/5E9AA7EB-C8D1-E111-A97D-002618943927.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/66346724-CAD1-E111-9F1B-0026189438D6.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/683B3346-C9D1-E111-AFC3-002618943838.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/6A46A729-C9D1-E111-A0D1-0026189438AA.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/6AA6C8B8-C7D1-E111-B621-001A928116F4.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/6E0252F5-C9D1-E111-B8A7-001BFCDBD190.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/544EA114-CAD1-E111-B9CF-0018F3D096C0.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/548CD2D7-C9D1-E111-977C-001A92811716.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/72E7FAF4-C4D1-E111-8EA1-002618943886.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/7A1FBE87-CAD1-E111-AEBE-002618943838.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/58DEDA2D-CAD1-E111-97AC-003048679180.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/5CBE9A4E-C9D1-E111-85A1-002354EF3BDC.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/7047D535-C9D1-E111-979E-002618943966.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/8CE81DAE-C8D1-E111-BAA6-002618943838.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/8EE3D74D-C7D1-E111-ADA0-003048678F74.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/7217E925-CBD1-E111-A39A-0026189438CB.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/9098DA0E-CAD1-E111-885B-0026189438D5.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/824E07AD-CAD1-E111-B612-002618943926.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/842F6556-CCD1-E111-87BD-002354EF3BE3.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/A4BD107B-CBD1-E111-8128-002354EF3BE6.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/A6E4C57B-C9D1-E111-8ECF-001A92810AA0.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/8678ADBB-C9D1-E111-B1C7-003048678B16.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/AE672405-CAD1-E111-9B59-003048678B34.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/8EE7F2BD-C9D1-E111-A2D5-001A92810AA8.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/B222C283-CBD1-E111-9DC1-00261894385A.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/B86F11BD-C7D1-E111-9F98-001BFCDBD166.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/C0BA63F2-C9D1-E111-A564-0018F3D09702.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/C4849ADA-C8D1-E111-9579-002618943905.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/90CDFD57-C9D1-E111-BB85-001A928116EE.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/C4EEAA63-C5D1-E111-B8CC-0026189438DE.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/C64C806F-CAD1-E111-B5F3-003048D3C010.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/9A65BD5E-C5D1-E111-A022-0026189438D6.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/A89A5272-CAD1-E111-A061-0018F3D096D8.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/D064B6BD-C9D1-E111-93EB-003048679180.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/D22DF865-C9D1-E111-8C2E-002618943800.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/B083CA5F-C9D1-E111-A226-0018F3D0960C.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/D8CD2169-C9D1-E111-AB6A-0018F3D096BC.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/DC324AC9-C8D1-E111-B6B4-0026189438E6.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/E09CC73C-C9D1-E111-B1C0-002618943863.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/C4D3D385-C7D1-E111-9F78-002618943862.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/F0A47BBA-CAD1-E111-AB67-003048678D86.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/F41939ED-CAD1-E111-B1ED-003048678FC6.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/F67641E6-C4D1-E111-901E-003048678B86.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/F6790673-C5D1-E111-B742-003048D15D22.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/F8339D38-CBD1-E111-BED4-0026189438DD.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/C810AB29-CAD1-E111-9061-00261894385A.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/CA866C8F-CBD1-E111-9094-00261894396E.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/D2780503-CAD1-E111-B154-00304867D446.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/EC39D895-CAD1-E111-B3C5-002618943800.root',
        '/store/relval/CMSSW_5_2_6-newconditions_RelVal_R198955_120719/MinimumBias/RAW/v3/0000/FCB7BA85-C9D1-E111-99D8-001A92811742.root' 
        ])
else:
    process.source.fileNames.extend([
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/D4C2D7A1-59A3-E111-8E62-003048D37694.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/CACD5368-57A3-E111-B292-003048F024DE.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/801A69EF-53A3-E111-BAC9-003048D2BB58.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/66D6CCE9-53A3-E111-87CD-002481E0D790.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/5A76C4CC-58A3-E111-AFED-001D09F27067.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/2E977848-52A3-E111-9EBF-BCAEC5329709.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/2E7E2B8C-51A3-E111-B818-001D09F28EA3.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/26038F94-54A3-E111-97A9-003048D2C0F0.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/0ED82FA8-51A3-E111-A534-003048F11114.root',
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/00124554-5AA3-E111-9004-001D09F2AD84.root',
        ])


##-- Output
process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands, #DQMEventContent
    fileName = cms.untracked.string('JetMET_DQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQM')
    )
)
if (dataset.find("MET")==0):
    process.DQMoutput.fileName = cms.untracked.string('JetMET_METData_DQM.root')
elif (dataset.find("JetMon")==0):
    process.DQMoutput.fileName = cms.untracked.string('JetMET_JetMonData_DQM.root')
elif (dataset.find("JetHT")==0):
    process.DQMoutput.fileName = cms.untracked.string('JetMET_JetHTData_DQM.root')
elif (dataset.find("reference")==0):
    process.DQMoutput.fileName = cms.untracked.string('JetMET_Ref_DQM.root')
elif (dataset.find("newcondition")==0):
    process.DQMoutput.fileName = cms.untracked.string('JetMET_NewCon_DQM.root')
else:
    process.DQMoutput.fileName = cms.untracked.string('JetMET_METData_DQM.root')

    
##-- Logger
process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical', 
        'cout')
)


##-- Config
from DQMOffline.Trigger.JetMETHLTOfflineSource_cfi import *
process.jetMETHLTOfflineSource.processname = cms.string("HLT")
process.jetMETHLTOfflineSource.triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT")
process.jetMETHLTOfflineSource.triggerResultsLabel = cms.InputTag("TriggerResults","","HLT")
process.jetMETHLTOfflineSource.plotEff = cms.untracked.bool(True)


##-- Let's it runs
process.JetMETSource_step = cms.Path( #process.L1T1coll *
                                      #process.noscraping *
                                      #process.primaryVertexFilter *
                                      process.jetMETHLTOfflineAnalyzer )
process.JetMETClient_step = cms.Path( process.jetMETHLTOfflineClient )
process.dqmsave_step      = cms.Path( process.dqmSaver )
process.DQMoutput_step    = cms.EndPath( process.DQMoutput )
# Schedule
process.schedule = cms.Schedule(process.JetMETSource_step,
                                process.JetMETClient_step,
                                process.dqmsave_step,
                                process.DQMoutput_step)
