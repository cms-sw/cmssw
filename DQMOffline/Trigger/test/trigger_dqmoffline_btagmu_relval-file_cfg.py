
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
process.load("DQMOffline.Trigger.BTagHLTOfflineSource_cfi")
process.load("DQMOffline.Trigger.BTagHLTOfflineClient_cfi")

from DQMOffline.Trigger.FourVectorHLTOfflineClient_cfi import *
from DQMOffline.Trigger.BTagHLTOfflineClient_cfi import *

process.hltclient = cms.Sequence(hltFourVectorClient)

hltFourVectorClient.prescaleLS = cms.untracked.int32(-1)
hltFourVectorClient.monitorDir = cms.untracked.string('')
hltFourVectorClient.prescaleEvt = cms.untracked.int32(1)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = 
	cms.untracked.vstring(
#    '/store/data/Run2010B/Jet/RECO/Nov4ReReco_v1/0040/52CEF71E-D8EB-DF11-9579-003048F0E7FC.root'
#    '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/E650DA99-1316-DE11-B057-000423D9A2AE.root'
    '/store/relval/CMSSW_3_11_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START311_V1_highstats-v1/0005/FAB506BD-922B-E011-BDE6-0026189438BA.root',
    '/store/relval/CMSSW_3_11_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START311_V1_highstats-v1/0005/ECD646B9-842B-E011-95E4-001A92971B48.root',
    
    )
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



process.psource = cms.Path(process.btagHLTOfflineSource*process.btagHLTOfflineClient)
process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


