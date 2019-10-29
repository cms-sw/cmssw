import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQM.HLTEvF.HLTMonitor_cff")

process.load("DQMServices.Core.DQM_cfg")

### include your reference file
process.DQMStore.referenceFileName = 'ref.root'

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    ### QCD CMSSW 3X
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/00670343-30E8-DD11-838D-000423D98BC4.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/1CEAEC44-32E8-DD11-A0A1-000423D98800.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/2E1704ED-6AE8-DD11-AFCF-001D09F24D67.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/3C64F3A9-22E8-DD11-9B0A-001617DBD332.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/46B94395-32E8-DD11-A37F-000423D6006E.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/5EA23D88-21E8-DD11-8544-000423DD2F34.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/629A7A7D-25E8-DD11-974D-000423D6CAF2.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/66BABB5C-31E8-DD11-8906-000423D99896.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/705B8646-36E8-DD11-ADD9-000423D6CA02.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/92EF21F9-22E8-DD11-9B77-001617DBD316.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/A85E9B7A-25E8-DD11-8E44-000423D6AF24.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/C4DFA74F-33E8-DD11-B1F5-001617E30E2C.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/D22FFF1F-26E8-DD11-8A08-001617E30CC8.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/D62F9B91-23E8-DD11-A738-000423D991F0.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/DABC2D7C-24E8-DD11-9D65-000423D6B42C.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/DC166450-22E8-DD11-9908-000423D94AA8.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/E8494F98-23E8-DD11-9C08-000423D9863C.root'
    )
)

###  DQM Source program (in DQMServices/Examples/src/DQMSourceExample.cc)
process.dqmSource   = cms.EDAnalyzer("DQMSourceExample",
        monitorName = cms.untracked.string('YourSubsystemName'),
        prescaleEvt = cms.untracked.int32(1),
        prescaleLS  =  cms.untracked.int32(1)                    
                                   )

### run the quality tests as defined in QualityTests.xml
from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/HLTEvF/test/JetMETQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    testInEventloop = cms.untracked.bool(True),
    verboseQT =  cms.untracked.bool(True)                 
                              )

#### BEGIN DQM Online Environment #######################
    
### replace YourSubsystemName by the name of your source ###
### use it for dqmEnv, dqmSaver
process.load("DQMServices.Components.DQMEnvironment_cfi")
#process.DQM.collectorHost = 'srv-c2d05-XX'
#process.DQM.collectorPort = 9190
### path where to save the output file
process.dqmSaver.dirName = '.'

### the filename prefix 
process.dqmSaver.producer = 'DQM'

### possible conventions are "Online", "Offline" and "RelVal"
process.dqmSaver.convention = 'Online'

process.dqmEnv.subSystemFolder = 'HLTJetMET'

### optionally change fileSaving  conditions
#process.dqmSaver.saveByLumiSection = -1
#process.dqmSaver.saveByMinute      = -1
#process.dqmSaver.saveByEvent       = -1
#process.dqmSaver.saveByRun         =  1
#process.dqmSaver.saveAtJobEnd      = False


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

#process.p = cms.EndPath(process.dqmSaver)
process.p = cms.EndPath(process.dqmSource*process.qTester*process.dqmEnv*process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

