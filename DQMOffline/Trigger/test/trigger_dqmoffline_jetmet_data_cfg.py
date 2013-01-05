##-- Starting
import FWCore.ParameterSet.Config as cms
process = cms.Process("DQM")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.EventContent.EventContent_cff')


##--
dataset = "MET" #MET,JetHT,TTbar


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

process.GlobalTag.globaltag = 'GR_R_61_V6::All'
if (dataset.find("TTbar")==0):
    process.GlobalTag.globaltag = 'START61_V8::All'
    
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = cms.string( autoCond[ 'com10' ] )


##-- L1
#process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
#process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
#process.L1T1coll=process.hltLevel1GTSeed.clone()
#process.L1T1coll.L1TechTriggerSeeding = cms.bool(True)
#process.L1T1coll.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')
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
    input = cms.untracked.int32(5000)                             
)
process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring())
if (dataset.find("MET")==0):
    process.source.fileNames.extend([
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/FCB36C6C-964C-E211-8BD2-002618943921.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/F25A74B3-964C-E211-BF1A-002618943911.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/F03B837A-914C-E211-855C-003048FFD744.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/EA0C8616-9B4C-E211-B101-0025905822B6.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/E86141DB-954C-E211-B7B2-0026189437F0.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/DACF0D94-924C-E211-B4CA-00248C55CC3C.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/D8C8D7BD-934C-E211-B51E-002618943880.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/D20BCB79-934C-E211-A20F-003048679080.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/D04078F2-944C-E211-B116-003048678ED4.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/CCFF6822-964C-E211-9BF2-003048678FD6.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/C208C147-974C-E211-AF65-00261894387E.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/BE026549-924C-E211-9CF3-002590596490.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/B43BD5B0-954C-E211-B15A-003048FFCC2C.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/B0417877-934C-E211-987C-003048FFCB74.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/B036C912-944C-E211-8B3D-002618943880.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/ACF951D5-984C-E211-BA21-003048678FDE.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/AA9C32C5-904C-E211-B310-00261894387A.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/A4679D9D-974C-E211-803E-00304867924E.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/A08D7A3A-A64C-E211-98F5-003048FFCC2C.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/A0215F34-984C-E211-A1B1-0026189437F8.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/9ED0019D-944C-E211-986D-00248C0BE018.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/9EBD1BED-974C-E211-BE33-003048678BAC.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/98134917-A04C-E211-9A93-0025905964BA.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/94D75DBD-934C-E211-A47C-002618943811.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/90BFE3FD-8E4C-E211-8822-003048D15E24.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/90812186-984C-E211-A97A-00261894387E.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/8A46C532-984C-E211-93C4-003048678EE2.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/889DE580-984C-E211-AC5E-003048679010.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/80341031-984C-E211-8F95-00248C0BE016.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/7C4CB3FA-994C-E211-839B-003048FFCC2C.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/7421B32A-934C-E211-B3A0-003048679296.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/729BFFF0-944C-E211-81D1-003048678BF4.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/6A420B92-994C-E211-BDAD-003048FFD736.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/68945618-964C-E211-857C-0026189438CE.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/6065FF1B-994C-E211-9A83-002618943971.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/5EB3B176-904C-E211-8DD8-002618943914.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/5E17F159-914C-E211-A8B4-002618943880.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/5C3CAA45-9C4C-E211-A0CA-003048FFD732.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/5AD63D2A-914C-E211-B29A-0025905964BA.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/58D9DFE9-974C-E211-B68B-002618FDA211.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/5882B7A6-914C-E211-9917-0026189438CC.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/52D0286D-994C-E211-ADE0-003048FFD744.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/5249F815-9E4C-E211-9470-00304867BFB2.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/50A48B4C-8F4C-E211-BD15-003048678F8A.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/4E0AA3EB-B04C-E211-A5EE-002618FDA204.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/4E058A58-944C-E211-BF18-003048FFD740.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/4C03EDC5-904C-E211-91B6-00261894393E.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/4AB0E2FB-914C-E211-8E0B-0026189438D9.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/4820B950-944C-E211-B75D-00261894397F.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/46337683-984C-E211-ACE2-00261894383C.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/383F998B-924C-E211-BCB7-003048FFD736.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/3821DDF7-914C-E211-845D-00304867BFB2.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/323A1DEE-984C-E211-81AC-00261894383E.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/3086F91C-994C-E211-A4D3-003048679296.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/2CC57764-984C-E211-8DF7-002590593872.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/1E3EAE25-9A4C-E211-AA24-00261894380B.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/1CDB9A96-A14C-E211-94D3-003048B95B30.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/142095EB-974C-E211-B0DE-0026189438E2.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/10A38E89-954C-E211-B019-0026189438E2.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/0ADA3755-954C-E211-9FB0-003048FFD740.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/0AD4239A-974C-E211-AF1E-002618943880.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/0A084265-8E4C-E211-B9EB-0026189438AF.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_met2012A/MET/RECO/v1/00000/08412EE7-984C-E211-9B85-0026189438C9.root',
        ])
elif (dataset.find("JetHT")==0):
    process.source.fileNames.extend([
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/FC31D58B-AE4C-E211-964C-00261894390A.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/FAA13D2C-D34C-E211-B467-00248C0BE018.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/F8DC0F22-A04C-E211-B63B-003048FFD754.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/F6F04AEB-A74C-E211-984B-003048678FF6.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/F6C5AB24-A74C-E211-9D17-0030486792B8.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/F274609C-A74C-E211-B982-0026189438ED.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/F0001723-A44C-E211-9DE6-00304867BEC0.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/EEACF76F-A94C-E211-B53C-0026189437EB.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/EC3F5D02-AD4C-E211-86A0-003048FFCB9E.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/E2BE3504-B84C-E211-976F-0026189438E8.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/E0501A21-C24C-E211-B217-0026189438E8.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/D84F6E25-AA4C-E211-ACA2-003048FF9AC6.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/D65F6528-B14C-E211-98B8-002618943913.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/D4B286C5-A64C-E211-B52C-0025905938AA.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/CCA38E03-A54C-E211-AC32-00304867BF18.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/CC9C348E-A34C-E211-B34F-00261894388F.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/CC097462-B94C-E211-8140-002618FDA26D.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/CAF00194-B04C-E211-9599-00304867D446.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/C8F5747F-AE4C-E211-9BCE-002618943973.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/C41DA852-A74C-E211-ADB8-0030486790A6.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/BC484644-AE4C-E211-9617-002590596468.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/BAA8E214-B64C-E211-86FE-002618943913.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/B8E4E20E-A04C-E211-A954-0026189437FE.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/B25A48FD-B34C-E211-BDAA-0026189438EB.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/AE782812-AF4C-E211-9B89-002618943921.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/ACE6DA9B-C04C-E211-A714-002618943970.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/AC21FAFC-B94C-E211-AB56-00261894398B.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/A8531503-B14C-E211-B279-00261894398D.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/A04A1E27-A44C-E211-9CBD-003048FFCB9E.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/9E6CF513-A74C-E211-8E33-0026189438ED.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/9AAB4156-A64C-E211-BBE1-003048FFD728.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/989FC175-A64C-E211-96AD-0030486792B8.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/94EE4915-A74C-E211-8F02-00304867902E.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/90CB45B3-CA4C-E211-A3BB-002618943972.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/8A852FDA-BF4C-E211-8199-0025905822B6.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/86993524-A14C-E211-9C26-003048FFCC18.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/866AE0CF-AB4C-E211-A1B6-003048FFCC1E.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/8612FDD3-9C4C-E211-A095-0026189438FD.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/82E90E7B-B14C-E211-86F1-003048679012.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/7EE4EE7B-AC4C-E211-B5DB-0026189438AA.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/767872F7-AD4C-E211-B3FA-002354EF3BDB.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/7656B57B-9F4C-E211-924A-003048F9EB46.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/7471DA81-A14C-E211-9A22-003048678BAC.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/74292C28-B54C-E211-B07A-00261894384F.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/6EFB9146-A14C-E211-9635-00304867BF18.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/6CFF208E-AA4C-E211-8AC7-003048FFD7A2.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/6CA48D5E-AF4C-E211-8DED-0026189438E8.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/68B44B99-A54C-E211-97B8-00261894391D.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/60C6361D-AF4C-E211-9164-00248C0BE005.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/60A00735-AB4C-E211-8239-003048FFCC0A.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/60219CCC-A84C-E211-A163-003048FFCBB0.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/5ABF7BC8-DA4C-E211-A165-0026189437FA.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/56EC7C4E-B74C-E211-BAA9-002618943842.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/56B8B0AE-AC4C-E211-B14D-003048FFD720.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/56AF18CB-9F4C-E211-823B-0030486790C0.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/502F8699-B24C-E211-98A5-003048D15DDA.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/4E570274-A44C-E211-9A3C-002618943973.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/4AA27D62-B24C-E211-9271-002618943974.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/467AE6EA-A14C-E211-B4D2-003048679188.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/4673A233-AE4C-E211-A146-002618943951.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/44DFC7B9-B14C-E211-AF1A-00261894386D.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/44556EBD-A04C-E211-AE38-0025905964C0.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/4056077D-AA4C-E211-B80C-00261894388A.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/3E921A43-A14C-E211-B074-0026189438E9.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/3C6BBAC8-9F4C-E211-B3C4-003048FFCB74.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/3A0A0E6D-BF4C-E211-B5D6-003048FF86CA.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/349A5143-AD4C-E211-B8EF-0025905964B2.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/3479381F-B14C-E211-86FD-003048678F8E.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/32EF928E-AA4C-E211-BBB9-0025905964CC.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/3239CAC3-A84C-E211-9F7E-00248C55CC9D.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/320699EE-AF4C-E211-8E2C-002354EF3BE4.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/308C8C15-B24C-E211-B98E-0026189438B8.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/2EB1978A-BE4C-E211-874B-0026189438E2.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/28415779-9F4C-E211-9364-0026189437FA.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/281BA0CF-C64C-E211-B1EF-002618943921.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/2471481B-A94C-E211-8582-003048FFD756.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/241EC76B-A04C-E211-8BFE-003048FFD736.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/240033B4-A04C-E211-9C20-00304867BF9A.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/22BEF367-A24C-E211-B653-003048F9EB46.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/2224C45E-A74C-E211-8930-003048678F8C.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/2031BFF5-A04C-E211-B29C-003048FFD7BE.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/1E7ABA64-A94C-E211-964F-003048FFD7D4.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/1CACF4AB-A24C-E211-8794-002618943947.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/1A77ABEA-A74C-E211-B69C-0030486790A6.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/1A25985C-B44C-E211-87C4-00261894398B.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/12F52AA6-AB4C-E211-A34C-003048678B36.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/12A3BD77-A84C-E211-BE37-002618943904.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/1249E41C-BE4C-E211-85F7-002618943978.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/0ED1698B-D24C-E211-955F-002618943810.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/089C96D7-B24C-E211-A909-00261894393E.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/087B7E26-B14C-E211-BFF3-0026189438DE.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/06F3D93F-A34C-E211-90EF-002618943950.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/065D81A4-AF4C-E211-AB6C-00261894387B.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/048DF902-A64C-E211-945A-0025905938B4.root',
        '/store/relval/CMSSW_6_1_0-GR_R_61_V6_RelVal_jet2012B/JetHT/RECO/v1/00000/0427FAE4-AB4C-E211-973C-002618943977.root',
        ])
elif (dataset.find("TTbar")==0):
    process.source.fileNames.extend([
        '/store/relval/CMSSW_6_1_0-START61_V8/RelValTTbar/GEN-SIM-RECO/v1/00000/AA60EFA5-E34C-E211-9F9D-0025905938A4.root',
        '/store/relval/CMSSW_6_1_0-START61_V8/RelValTTbar/GEN-SIM-RECO/v1/00000/92FED5E2-E44C-E211-A1F1-003048FFD71A.root',
        '/store/relval/CMSSW_6_1_0-START61_V8/RelValTTbar/GEN-SIM-RECO/v1/00000/42FAF5C9-EB4C-E211-A962-002618943960.root',
        ])
else:
    process.source.fileNames.extend([
        '/store/data/Run2012B/MET/RECO/PromptReco-v1/000/194/429/D4C2D7A1-59A3-E111-8E62-003048D37694.root'
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
#if (dataset.find("MET")==0):
#    process.DQMoutput.fileName = cms.untracked.string('JetMET_METData_DQM.root')
#elif (dataset.find("JetMon")==0):
#    process.DQMoutput.fileName = cms.untracked.string('JetMET_JetMonData_DQM.root')
#elif (dataset.find("JetHT")==0):
#    process.DQMoutput.fileName = cms.untracked.string('JetMET_JetHTData_DQM.root')
#elif (dataset.find("reference")==0):
#    process.DQMoutput.fileName = cms.untracked.string('JetMET_Ref_DQM.root')
#elif (dataset.find("newcondition")==0):
#    process.DQMoutput.fileName = cms.untracked.string('JetMET_NewCon_DQM.root')
#else:
#    process.DQMoutput.fileName = cms.untracked.string('JetMET_METData_DQM.root')

    
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
