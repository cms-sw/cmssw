import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQM/L1TMonitor/L1TEmulatorMonitor_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#off-line
process.GlobalTag.globaltag = 'GR10_P_V4::All'

#ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC, LTC,GMT,GLT 
#process.l1compare.COMPARE_COLLS = [0,0,0,0, 0,0,0,1,0, 0,0,0]
#process.l1compare.COMPARE_COLLS = [1,1,1,1, 1,1,1,0,1, 0,1,1]
process.l1compare.DumpMode = -1
process.l1compare.VerboseFlag = 1
process.l1demon.VerboseFlag = 1
#process.l1demonecal.VerboseFlag = 1
#process.l1demongct.VerboseFlag = 1
#process.l1tderct.verbose = False
process.l1demon.disableROOToutput = False
process.l1demonecal.disableROOToutput = False
process.l1demongct.disableROOToutput = False
process.l1tderct.disableROOToutput = False
#process.l1demon.RunInFilterFarm=True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


##to access new runs in castor, use dbs eg (specify desired run number in 'run=')
##https://cmsweb.cern.ch/dbs_discovery/getLFN_txt?dbsInst=cms_dbs_prod_global&blockName=*&dataset=/Cosmics/Commissioning09-v3/RAW&userMode=user&run=118969&what=cff
#check rfdir /castor/cern.ch/cms/store/data/Commissioning09/Cosmics/RAW/v3/000/118/969/FEA12E2E-EEC5-DE11-8BD0-000423D98C20.root
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/Commissioning10/Cosmics/RAW/v4/000/131/872/4CE9AD33-C736-DF11-B9D7-000423D952C0.root'
    )
)

process.outputEvents = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testdemon.root')
)
process.ep = cms.EndPath(process.outputEvents)
