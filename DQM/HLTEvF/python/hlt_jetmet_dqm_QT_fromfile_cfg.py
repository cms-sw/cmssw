import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
#process.load("DQM.HLTEvF.HLTMonitor_cff")

#process.load("DQM.HLTEvF.HLTMonJetMET_cfi")

#process.load("DQM.HLTEvF.HLTMonJetMET_E31_cfi")
#from DQM.HLTEvF.HLTMonJetMET_E31_cfi import *

process.load("DQM.HLTEvF.HLTMonJetMET_E28_cfi")
from DQM.HLTEvF.HLTMonJetMET_E28_cfi import *

##@$process.load("DQM.HLTEvF.jetmetDQMConsumer_cfi")
##@$from DQM.HLTEvF.jetmetDQMConsumer_cfi import *


process.load("DQMServices.Core.DQM_cfg")

### include your reference file
###process.DQMStore.referenceFileName = 'ref.root'

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(100)
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    ### QCD CMSSW 3X
        'file:/afs/cern.ch/user/j/jabeen/public/CMSSW_3_1_0_pre10/src/Test.root'
#        '/store/data/Commissioning09/Monitor/RAW/v1/000/082/548/A06226EF-4A30-DE11-A607-000423D94AA8.root'
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/F668AADD-4F16-DE11-96A4-001617C3B70E.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/D8DA8359-0616-DE11-8B00-000423D98AF0.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/CC17AAF2-0616-DE11-8C55-000423D6CA42.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/C87CCC26-4F16-DE11-A5E7-000423D98844.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/70820DE5-0616-DE11-9A9D-000423D986A8.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/6EA9847C-5A16-DE11-82B8-001617C3B6C6.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/6C075A5D-2016-DE11-BB66-000423D6CA72.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/6620613C-4F16-DE11-A033-000423D95220.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/5688EBA3-0516-DE11-8170-000423D94AA8.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/3E1BCACC-0516-DE11-B6DC-000423D6A6F4.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/364F41D9-0516-DE11-BF0A-000423D9853C.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/1C94B51A-AC16-DE11-8F94-001617C3B76E.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/1A3FDC02-0916-DE11-9DC7-001617E30D0A.root',
#       '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0003/00DD82EE-0616-DE11-ABAD-000423D6B42C.root'
#    '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar_cfi/GEN-SIM-DIGI-RECO/STARTUP_30X_FastSim_v1/0001/0C384E22-4116-DE11-9D9D-0018F3D09698.root',
    ##'/store/relval/CMSSW_3_1_0_pre4/RelValTTbar_cfi/GEN-SIM-DIGI-RECO/STARTUP_30X_FastSim_v1/0001/0E152E43-3E16-DE11-9BF7-001731AF6AE7.root'
    #'file:/uscms_data/d2/jabeen/work/CMSSW_3_1_0_pre4_jn/src/DQM/HLTEvF/python/0C384E22-4116-DE11-9D9D-0018F3D09698.root'

    )
)

###  DQM Source program (in DQMServices/Examples/src/DQMSourceExample.cc)
###process.dqmSource   = cms.EDAnalyzer("DQMSourceExample",
###        monitorName = cms.untracked.string('YourSubsystemName'),
###        prescaleEvt = cms.untracked.int32(1),
###        prescaleLS  =  cms.untracked.int32(1)                    
###                                   )

### run the quality tests as defined in QualityTests.xml
#@#process.qTester = cms.EDAnalyzer("QualityTester",
#@#    qtList = cms.untracked.FileInPath('DQM/HLTEvF/python/JetMETQualityTests.xml'),
#@#    prescaleFactor = cms.untracked.int32(1),                               
#@#    testInEventloop = cms.untracked.bool(True),
#@#    verboseQT =  cms.untracked.bool(True)                 
#@#                              )

#### BEGIN DQM Online Environment #######################
    
### replace YourSubsystemName by the name of your source ###
### use it for dqmEnv, dqmSaver
process.load("DQMServices.Components.DQMEnvironment_cfi")

##4 lines below are necessary to pick up right conditions in which data was taken.?????

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRAFT_31X::All'

#process.DQM.collectorHost = 'srv-c2d05-XX'
#process.DQM.collectorPort = 9190
### path where to save the output file
process.dqmSaver.dirName = '.'

### the filename prefix 
process.dqmSaver.producer = 'DQM'

### possible conventions are "Online", "Offline" and "RelVal"
process.dqmSaver.convention = 'Online'

process.dqmEnv.subSystemFolder = 'HLTMonJetMET'

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


#with consumer 

#process.p = cms.EndPath(process.hltMonJetMET*process.qTester*process.dqmEnv*process.dqmSaver)
process.p = cms.EndPath(process.hltMonJetMET*process.dqmEnv*process.dqmSaver)

##@$process.p = cms.EndPath(process.hltMonJetMET*process.jetmetDQMConsumer*process.qTester*process.dqmEnv*process.dqmSaver)




process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

