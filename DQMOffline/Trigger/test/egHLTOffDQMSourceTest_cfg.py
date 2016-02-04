import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMTest")

#set DQM enviroment
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#load and setup E/g HLT Offline DQM module
process.load("DQMOffline.Trigger.EgHLTOfflineSource_cfi")

#load calo geometry
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
# Other statements
process.GlobalTag.globaltag = 'STARTUP3X_V15::All'

#configure message logger to something sane
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 1000000000

process.MessageLogger.cerr.FwkSummary = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(500),
    limit = cms.untracked.int32(10000000)
)
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(500),
    limit = cms.untracked.int32(10000000)
)


process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(),
)
process.source.fileNames=[#"file:/media/usbdisk1/beamData09/122/314/02E8544C-70D8-DE11-85CF-001617C3B66C.root",
#"file:/media/usbdisk1/beamData09/122/314/4CAB3B6C-6BD8-DE11-845C-000423D9890C.root",
#"file:/media/usbdisk1/beamData09/122/314/7C7495C2-71D8-DE11-ACF2-001D09F248F8.root",
#"file:/media/usbdisk1/beamData09/122/314/9264A28F-87D8-DE11-83A1-001D09F24763.root",
#"file:/media/usbdisk1/beamData09/122/314/EE7B1AC4-6CD8-DE11-97BB-0030487A1FEC.root",
#"file:/media/usbdisk1/beamData09/122/314/F4387297-74D8-DE11-996C-001D09F24F1F.root",
#"file:/media/usbdisk1/beamData09/122/314/F62B040F-6CD8-DE11-9007-001D09F24664.root"]
   '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/BED3668B-85ED-DE11-A862-00261894396C.root', ]  
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_1.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_10.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_11.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_12.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_13.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_14.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_2.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_3.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_4.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_5.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_6.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_7.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_8.root",
#"file:/media/usbdisk1/beamData09/skims/123596/ExpressPhysics_TT40_41_BX_LS/tt_bit40_41_lumi_bx_skim_9.root"]


process.maxEvents = cms.untracked.PSet(
  
    input = cms.untracked.int32(-1)
)


process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")    
process.load("Configuration.EventContent.EventContent_cff")


process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RECO')),
    fileName = cms.untracked.string('test/relVal_350pre2Test.root')
)
process.FEVT.outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_DQMTest')


#monitor elements are converted to EDM format to store in CMSSW file
#client will convert them back before processing
process.psource = cms.Path(process.egHLTOffDQMSource*process.hltTrigReport*process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.FEVT)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''

