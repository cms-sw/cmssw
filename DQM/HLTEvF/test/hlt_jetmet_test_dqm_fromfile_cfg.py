import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQM.HLTEvF.HLTMonitor_cff")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    ### TTbar 3
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/1C51253D-D4CC-DD11-89EC-001D09F23A6B.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/26FB4A24-D2CC-DD11-845E-0019B9F72F97.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/30EE04C0-CACC-DD11-96CF-000423D99658.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/362BA44E-D2CC-DD11-8BAB-001617DBD5B2.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/6459EFC9-D7CC-DD11-8D80-001617DBCF6A.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/6AD9D6F6-D3CC-DD11-909A-001617C3B79A.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/7890DAB6-CDCC-DD11-968C-000423DD2F34.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/ACCC88FA-CFCC-DD11-AE2D-001D09F24637.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/ACFCB5C4-BCCC-DD11-832F-001D09F253C0.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/B2A7BEBE-CCCC-DD11-824A-000423D98E54.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/C2BBDEF8-CBCC-DD11-A8A0-001617DBD5AC.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/C65BB6E9-59CD-DD11-95F1-0030487A1990.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/D2BB240F-D3CC-DD11-9921-001617C3B5E4.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/DA78BADE-BCCC-DD11-AE39-0030487BC68E.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/DAA706E0-BDCC-DD11-AD9E-001D09F2432B.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/DC28FB7D-D5CC-DD11-A689-001D09F231C9.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/E02F1FE1-D4CC-DD11-AF0C-001D09F231C9.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/EAB4B18D-CFCC-DD11-B952-0030487C608C.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/FAE75F4C-C0CC-DD11-97B1-001617C3B706.root',
    '/store/relval/CMSSW_3_0_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0004/FC1FE58F-CCCC-DD11-9198-001617DBCF90.root'
    )
)

process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
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

process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

