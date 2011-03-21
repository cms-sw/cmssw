import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQMServices.Components.DQMEnvironment_cfi")


process.load("DQMOffline.Trigger.EgHLTOfflineSource_cfi")
process.load("DQMOffline.Trigger.EgHLTOfflineClient_cfi")


#load calo geometry
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")    
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input =cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(),
)
process.source.fileNames = [
    
    #-------/RelValZEE/CMSSW_3_11_1_hltpatch1-L1HLTST311_V0-v1/GEN-SIM-RECO-----9000evts-----------------------
    '/store/relval/CMSSW_3_11_1_hltpatch1/RelValZEE/GEN-SIM-RECO/L1HLTST311_V0-v1/0017/2404523C-9E41-E011-95E5-002618943886.root',
    '/store/relval/CMSSW_3_11_1_hltpatch1/RelValZEE/GEN-SIM-RECO/L1HLTST311_V0-v1/0016/B4BBD50B-5441-E011-8CC0-00248C55CC62.root',
    '/store/relval/CMSSW_3_11_1_hltpatch1/RelValZEE/GEN-SIM-RECO/L1HLTST311_V0-v1/0016/84CE6498-5041-E011-A8F9-002618943886.root',
    '/store/relval/CMSSW_3_11_1_hltpatch1/RelValZEE/GEN-SIM-RECO/L1HLTST311_V0-v1/0016/70D3FE8E-5141-E011-95F5-001A928116E8.root',
    '/store/relval/CMSSW_3_11_1_hltpatch1/RelValZEE/GEN-SIM-RECO/L1HLTST311_V0-v1/0016/102A8E2C-5141-E011-A8E3-002618943886.root'
     #---------/RelValWE/CMSSW_3_11_1_hltpatch1-L1HLTST311_V0-v1/GEN-SIM-RECO----9000evts-----------------------
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValWE/GEN-SIM-RECO/L1HLTST311_V0-v1/0017/58808A1C-9E41-E011-B316-001A928116BA.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValWE/GEN-SIM-RECO/L1HLTST311_V0-v1/0016/B6A508EA-5641-E011-9386-0030486790BA.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValWE/GEN-SIM-RECO/L1HLTST311_V0-v1/0016/A4564535-4E41-E011-B62D-001A928116EE.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValWE/GEN-SIM-RECO/L1HLTST311_V0-v1/0016/74FB9618-4F41-E011-A6C7-00261894393A.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValWE/GEN-SIM-RECO/L1HLTST311_V0-v1/0016/06381E0B-4D41-E011-9C22-002618943876.root'
    #--------/RelValLM9p/CMSSW_3_11_1_hltpatch1-L1HLTST311_V0_highstats-v1/GEN-SIM-RECO---25000evts------------------
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0018/4CFE7435-5F42-E011-987D-003048678E6E.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/FAA0513D-9441-E011-9FDE-001A92971BCA.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/ECEE0C9F-9641-E011-BB74-0018F3D09678.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/E0A71151-9741-E011-848B-003048678E2A.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/D8C02EB7-9441-E011-93B5-002618943826.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/C81B84A1-9B41-E011-B074-003048D15DB6.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/C22B3C27-9541-E011-BDD3-00261894388A.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/B8824A31-9441-E011-A8F0-002618943880.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/B63DCBD2-9A41-E011-8ED6-00304867906C.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/B046E80A-9D41-E011-9138-003048678BAE.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/AE3FE8A2-9541-E011-A69B-002618943948.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/86CA9326-9541-E011-A6E5-002618943948.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/7CB35BA7-9541-E011-8FE9-002618943877.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/76A39BFE-A041-E011-8A28-003048D15DB6.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/54E509FF-9E41-E011-8A22-0026189438A9.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/52918604-A541-E011-8CC3-003048D3FC94.root',
    #'/store/relval/CMSSW_3_11_1_hltpatch1/RelValLM9p/GEN-SIM-RECO/L1HLTST311_V0_highstats-v1/0017/28D28F46-9841-E011-B55A-001A92810AC0.root'
    #-------------------------------------
    #'file:/media/usbdisk1/ZeeRelVal_311_FC71916C-756B-DE11-8631-000423D94700.root'
    ]

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    debugInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('EgammaHLTOffline','EgHLTOfflineClient'),
    destinations = cms.untracked.vstring('debugInfo', 
        'detailedInfo', 
        'critical', 
        'cout')
 )

process.psource = cms.Path(process.egHLTOffDQMSource*process.egHLTOffDQMClient)
process.p = cms.EndPath(process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
#process.GlobalTag.globaltag = 'STARTUP::All'
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
#---Morse-----
process.dqmSaver.dirName = '/data/ndpc3/c/dmorse/HLTDQMrootFiles'
#-------


