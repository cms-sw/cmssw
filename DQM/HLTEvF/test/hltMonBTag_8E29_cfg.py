import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# /RelValTTbar/CMSSW_3_3_0_pre6-STARTUP31X_V8-v1/GEN-SIM-DIGI-RAW-HLTDEBUG 
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0006/D2067865-17B1-DE11-905E-001D09F29146.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/F2DA3881-FBB0-DE11-B713-001D09F24FEC.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/EAEF920F-FCB0-DE11-981C-001D09F25109.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/EAEBDB17-F5B0-DE11-9432-001D09F276CF.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/D0B6983A-F6B0-DE11-A4F0-001D09F2B2CF.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/C295B70E-FBB0-DE11-8A80-001D09F29146.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/C2896600-FDB0-DE11-BCBC-001D09F24FEC.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/BA280062-FCB0-DE11-8E85-000423D99AA2.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/B6F86812-FEB0-DE11-9034-000423D991F0.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/B2887EB6-F4B0-DE11-9466-000423D60FF6.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/AA7BEB65-FCB0-DE11-982C-001D09F2AD4D.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/8A3E5F52-F9B0-DE11-B381-001D09F28F1B.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/8078368B-FAB0-DE11-9CE3-001D09F2B2CF.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/7EB91F7C-FAB0-DE11-A3B3-000423D99AA2.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/7054BAA5-F4B0-DE11-BF9B-000423D99614.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/6AC09EEE-F2B0-DE11-890A-000423D94E70.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/64E9F94A-F5B0-DE11-9370-000423D33970.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/44B5986F-FEB0-DE11-83B5-000423D991F0.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/442F1CD6-F9B0-DE11-BEB2-001D09F24353.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/20BD8F19-F7B0-DE11-9DBF-001D09F28D4A.root',
        '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0005/00655C7F-FBB0-DE11-B0A9-001D09F2D426.root'
    )
)

process.load("DQMServices.Core.DQM_cfg")

import DQM.HLTEvF.hltMonBTagIPSource_cfi
import DQM.HLTEvF.hltMonBTagMuSource_cfi
import DQM.HLTEvF.hltMonBTagIPClient_cfi
import DQM.HLTEvF.hltMonBTagMuClient_cfi

# definition of the Sources for 8E29
process.hltMonBTagIP_Jet50U_Source = DQM.HLTEvF.hltMonBTagIPSource_cfi.hltMonBTagIPSource.clone()
process.hltMonBTagIP_Jet50U_Source.storeROOT = True

process.hltMonBTagMu_Jet10U_Source = DQM.HLTEvF.hltMonBTagMuSource_cfi.hltMonBTagMuSource.clone()
process.hltMonBTagMu_Jet10U_Source.storeROOT = True

process.hltMonBTagSource_8E29 = cms.Sequence( process.hltMonBTagIP_Jet50U_Source + process.hltMonBTagMu_Jet10U_Source )

# definition of the Clients for 8E29
process.hltMonBTagIP_Jet50U_Client = DQM.HLTEvF.hltMonBTagIPClient_cfi.hltMonBTagIPClient.clone()
process.hltMonBTagIP_Jet50U_Client.updateRun = True
process.hltMonBTagIP_Jet50U_Client.storeROOT = True

process.hltMonBTagMu_Jet10U_Client = DQM.HLTEvF.hltMonBTagMuClient_cfi.hltMonBTagMuClient.clone()
process.hltMonBTagMu_Jet10U_Client.updateRun = True
process.hltMonBTagMu_Jet10U_Client.storeROOT = True

process.hltMonBTagClient_8E29 = cms.Sequence( process.hltMonBTagIP_Jet50U_Client + process.hltMonBTagMu_Jet10U_Client )

process.dqm = cms.Path( process.hltMonBTagSource_8E29 + process.hltMonBTagClient_8E29 )
