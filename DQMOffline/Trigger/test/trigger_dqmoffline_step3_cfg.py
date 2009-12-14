import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("Configuration.StandardSequences.Geometry_cff")
#
#  DQMOffline
#
process.load("DQMOffline.Configuration.DQMOffline_SecondStep_cff")
#process.load("DQMOffline.Trigger.FourVectorHLTOfflineClient_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
 fileMode = cms.untracked.string('FULLMERGE')
)

process.source = cms.Source("PoolSource",
#    dropMetaData = cms.untracked.bool(True),
    processingMode = cms.untracked.string("RunsLumisAndEvents"),
    fileNames = cms.untracked.vstring(
#'file:myEDM.root'
#'file:myDQMOfflineTriggerEDM.root'
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_1.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_2.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_3.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_4.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_5.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_6.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_7.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_8.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_9.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_10.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_11.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_12.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_13.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_14.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_15.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_16.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_17.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_18.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_19.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_20.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_21.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_22.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_23.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_24.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_25.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_26.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_27.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_28.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_29.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_30.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_31.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_32.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_33.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_34.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_35.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_36.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_37.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_38.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_39.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_40.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_41.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_42.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_43.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_44.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_45.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_46.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_47.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_48.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_49.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_50.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_51.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_52.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_53.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_54.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_55.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_56.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_57.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_58.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_59.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_60.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_61.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_62.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_63.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_64.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_65.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_66.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_67.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_68.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_69.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_70.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_71.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_72.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_73.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_74.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_75.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_76.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_77.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_78.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_79.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_80.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_81.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_82.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_83.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_84.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_85.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_86.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_87.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_88.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_89.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_90.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_91.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_92.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_93.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_94.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_95.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_96.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_97.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_98.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_99.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_100.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_101.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_102.root',
      'rfio:/castor/cern.ch/user/r/rekovic/DQM_StreamExpress/test/124120/myDQMOfflineTriggerEDM_103.root'
      
    )
)

process.source.processingMode = "RunsAndLumis"

process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/StreamExpress/BeamCommissioning09-v8/DQMOffline'

process.DQMStore.collateHistograms = False
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun = False

process.p1 = cms.Path(process.EDMtoMEConverter*process.hltFourVectorClient*process.dqmSaver)
#process.p1 = cms.Path(process.EDMtoMEConverter*process.triggerOfflineDQMClient * process.hltOfflineDQMClient * process.dqmSaver)
#process.p1 = cms.Path(process.EDMtoMEConverter*process.dqmSaver)

