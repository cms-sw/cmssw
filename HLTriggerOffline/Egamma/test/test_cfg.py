import FWCore.ParameterSet.Config as cms

process = cms.Process("dqm")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/043C13EF-716C-DD11-B1C4-000423D6CA42.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/1CC14020-716C-DD11-89CA-000423D99F3E.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/2C934FEE-706C-DD11-9248-000423D94534.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/321D842A-7A6C-DD11-A010-001617C3B64C.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/5C123284-716C-DD11-837E-000423D6CA42.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/68A6DC9C-726C-DD11-AC0C-001617E30D00.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/8243AD53-7A6C-DD11-9915-000423D99BF2.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/8EAEA95B-776C-DD11-8361-001617E30CC8.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/CA94E86F-7A6C-DD11-B3F9-001617C3B6CC.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/CAED580B-7A6C-DD11-8B5C-001617E30D4A.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/CC1CC0A4-7A6C-DD11-9462-000423D95220.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/F0949867-7A6C-DD11-A6A9-000423D98634.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/FC48C92B-7A6C-DD11-A5C5-001617C3B706.root', 
        '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0005/B25CA5CA-8A6C-DD11-8884-000423D6B358.root')
)

process.load("HLTriggerOffline.Egamma.EgammaValidation_cff")
process.testW = cms.Path(process.egammavalWenu)
process.testZ = cms.Path(process.egammavalZee)

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.p = cms.EndPath(process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
