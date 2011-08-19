
import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

#
#  DQM SERVICES
#
process.load("DQMServices.Core.DQM_cfg")

#
#  DQM SOURCES
#
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQMOffline.Trigger.FourVectorHLTOffline_cfi")
process.load("DQMOffline.Trigger.FourVectorHLTOfflineClient_cfi")
process.load("DQMOffline.Trigger.JetMETHLTOfflineSource_cfi")
process.load("DQMOffline.Trigger.JetMETHLTOfflineClient_cfi")
#process.load("DQMOffline.Trigger.L1TMonitor_dqmoffline_cff")
#process.load("DQMOffline.Trigger.Tau.HLTTauDQMOffline_cff")
#process.load("DQMOffline.Trigger.EgammaHLTOffline_cfi")
#process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

from DQMOffline.Trigger.FourVectorHLTOfflineClient_cfi import *
from DQMOffline.Trigger.JetMETHLTOfflineClient_cfi import *

process.hltclient = cms.Sequence(hltFourVectorClient)

hltFourVectorClient.prescaleLS = cms.untracked.int32(-1)
hltFourVectorClient.monitorDir = cms.untracked.string('')
hltFourVectorClient.prescaleEvt = cms.untracked.int32(1)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = 
	#cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_31X_v1/0002/F2DD152F-2D33-DE11-B6F4-000423D9870C.root')
	#cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0002/F4802588-F232-DE11-A617-000423D94524.root')
	#cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_A_v1/0000/C69ACB56-532B-DE11-870F-001617C3B69C.root')
	cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/E650DA99-1316-DE11-B057-000423D9A2AE.root')
	#cms.untracked.vstring('file:test.root')
        #cms.untracked.vstring('/store/relval/CMSSW_2_1_7/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/0C3B40D7-F87D-DD11-A9FB-000423D998BA.root','/store/relval/CMSSW_2_1_7/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/3A5455F3-F87D-DD11-AEF4-000423D94534.root')
)

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



process.psource = cms.Path(process.jetMETHLTOfflineSource*process.jetMETHLTOfflineClient)
process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


