import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
#----------------------------
# Event Source
#-----------------------------
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring("*")
)
process.EventStreamHttpReader.maxEventRequestRate = cms.untracked.double(25.0)   
#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Core.DQM_cfg")
process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1t_reference.root"
process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.test.environment_cfi")

#-----------------------------
#
#  DQM SOURCES
#
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("DQM.L1TMonitor.L1TMonitor_cff")

process.load("DQM.L1TMonitorClient.L1TMonitorClient_cff")
process.load("DQM.TrigXMonitor.L1Scalers_cfi")
process.load("DQM.TrigXMonitorClient.L1TScalersClient_cfi")
process.l1s.l1GtData = cms.InputTag("l1GtUnpack","","DQM")
process.l1s.dqmFolder = cms.untracked.string("L1T/L1Scalers_SM") 
process.l1tsClient.dqmFolder = cms.untracked.string("L1T/L1Scalers_SM")
process.p3 = cms.EndPath(process.l1s+process.l1tsClient)

#process.l1GtParameters.BstLengthBytes = 52
##  Available data masks (case insensitive):
##    all, gt, muons, jets, taujets, isoem, nonisoem, met
process.l1tEventInfoClient.dataMaskedSystems = cms.untracked.vstring("Muons","Jets","TauJets","IsoEm","NonIsoEm","MET")

##  Available emulator masks (case insensitive):
##    all, dttf, dttpg, csctf, csctpg, rpc, gmt, ecal, hcal, rct, gct, glt
process.l1tEventInfoClient.emulatorMaskedSystems = cms.untracked.vstring("All")


process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')

process.EventStreamHttpReader.consumerName = 'L1T DQM Consumer'
process.dqmEnv.subSystemFolder = 'L1T'

#process.hltMonScal.remove("l1tscalers")
