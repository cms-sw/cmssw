import FWCore.ParameterSet.Config as cms

process = cms.Process("HEAVYFLAVORVALIDATION")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0004/42E21769-32A2-DE11-A54F-00304867915A.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0003/A0B6EE13-CAA1-DE11-BE65-001A9281171E.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0003/9C0C4FF0-CCA1-DE11-91B6-001A92810AD4.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0003/2668C475-C9A1-DE11-A075-0018F3D096CA.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0003/1A413ED3-CAA1-DE11-8856-0018F3D09676.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0003/04965931-CCA1-DE11-A28F-001A92971BDA.root'
  ),
  secondaryFileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0004/16618E4A-32A2-DE11-A323-001A928116E0.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/EA04DED1-CAA1-DE11-9A4F-001731AF68B9.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/D445FDEE-CCA1-DE11-AFA9-0018F3D096DC.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/AE8C8DD1-CAA1-DE11-B556-0018F3D09678.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/8E050334-CCA1-DE11-B41A-001BFCDBD11E.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/64C1F508-C8A1-DE11-9C14-001731A2870D.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/627A477A-CBA1-DE11-B029-001A92971B04.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/58B3FA67-C9A1-DE11-BD62-0018F3D0970E.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/5618CD0E-CAA1-DE11-B213-0018F3D0960A.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/5262BD3F-CCA1-DE11-B1CD-0018F3D0960E.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/4081C310-CAA1-DE11-9977-001A92971BB8.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/32FA0870-C9A1-DE11-BB55-0018F3D09710.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/2EA0DCEC-CCA1-DE11-93FD-0018F3D0967E.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/2698FCCF-CAA1-DE11-87FD-001A92971AA4.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/1A37CBD2-CAA1-DE11-931B-001731A28BE1.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/149D3E77-CBA1-DE11-A028-001BFCDBD11E.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/10DBC738-CEA1-DE11-8C63-001731EF61B4.root'
  )
)
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1000)
)

#process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.debugModules = cms.untracked.vstring('heavyFlavorValidation')
#
#process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
#process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )

process.load("HLTriggerOffline.HeavyFlavor.heavyFlavorValidationSequence_cff")
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load("DQMServices.Components.DQMStoreStats_cfi")
process.path = cms.Path(
  process.heavyFlavorValidationSequence
  *process.endOfProcess*process.dqmStoreStats
)

process.outputmodule = cms.OutputModule("PoolOutputModule",
  outputCommands = cms.untracked.vstring(
    'drop *', 
    'keep *_MEtoEDMConverter_*_HEAVYFLAVORVALIDATION'
  ),
  fileName = cms.untracked.string('/tmp/heavyFlavorValidation.root')
)
process.endpath = cms.EndPath(process.outputmodule)

