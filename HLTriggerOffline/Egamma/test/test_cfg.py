import FWCore.ParameterSet.Config as cms

process = cms.Process("dqm")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
##process.source = cms.Source("PoolSource",
##     fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/043C13EF-716C-DD11-B1C4-000423D6CA42.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/1CC14020-716C-DD11-89CA-000423D99F3E.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/2C934FEE-706C-DD11-9248-000423D94534.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/321D842A-7A6C-DD11-A010-001617C3B64C.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/5C123284-716C-DD11-837E-000423D6CA42.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/68A6DC9C-726C-DD11-AC0C-001617E30D00.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/8243AD53-7A6C-DD11-9915-000423D99BF2.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/8EAEA95B-776C-DD11-8361-001617E30CC8.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/CA94E86F-7A6C-DD11-B3F9-001617C3B6CC.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/CAED580B-7A6C-DD11-8B5C-001617E30D4A.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/CC1CC0A4-7A6C-DD11-9462-000423D95220.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/F0949867-7A6C-DD11-A6A9-000423D98634.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0004/FC48C92B-7A6C-DD11-A5C5-001617C3B706.root', 
##         '/store/relval/CMSSW_2_1_4/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V5_v1/0005/B25CA5CA-8A6C-DD11-8884-000423D6B358.root')
## )
## process.source = cms.Source("PoolSource",
##                             fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_10/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/12B3680D-5B99-DD11-8662-000423D944FC.root',
##                                                               '/store/relval/CMSSW_2_1_10/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/9A8F5C35-FD99-DD11-9DA7-000423D99658.root',
##                                                               '/store/relval/CMSSW_2_1_10/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/E415E155-5B99-DD11-9250-000423D986C4.root')
##                             )

process.load("HLTriggerOffline.Egamma.EgammaValidation_cff")
process.post=cms.EDAnalyzer("EmDQMPostProcessor",
                            subDir = cms.untracked.string("HLT/HLTEgammaValidation")                   
    )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/02E306BF-9599-DD11-8EDB-000423D98834.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/108AC9BB-9499-DD11-9304-001617C3B710.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/1A8E2A55-9699-DD11-B29A-000423D6B358.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/28C21BAE-9699-DD11-B7E9-001617DBCF90.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/28E618A4-9399-DD11-A1A5-000423D98C20.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/2AA2D050-9999-DD11-A234-000423D98BC4.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/366308A6-9999-DD11-B5D6-000423D99896.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/3C012B6E-9799-DD11-8279-000423D99EEE.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/5E764511-9899-DD11-AF11-001617E30F4C.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/7A7AD63A-9599-DD11-8C84-0019DB29C620.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/CE9EFEFC-9699-DD11-93AA-001617C3B76A.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/DE7FB293-9799-DD11-AB92-000423D6CA6E.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/E60356A4-9899-DD11-897A-001617E30D52.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/EC4A185E-9899-DD11-8869-000423D6B5C4.root',
                                         '/store/relval/CMSSW_2_1_10/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/F0DE7E3B-FD99-DD11-9457-000423D99660.root'
                                         )
                            )

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.p = cms.EndPath(process.post+process.dqmSaver)

process.testW = cms.Path(process.egammaValidationSequence)

process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
