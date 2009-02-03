import FWCore.ParameterSet.Config as cms

process = cms.Process("HEAVYFLAVORVALIDATION")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-RECO/STARTUP_30X_v1/0001/12791DD7-70E9-DD11-887B-003048679188.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-RECO/STARTUP_30X_v1/0001/1847F131-73E9-DD11-86F2-0017312B577F.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-RECO/STARTUP_30X_v1/0001/3A829C29-6EE9-DD11-B933-001731AF6865.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-RECO/STARTUP_30X_v1/0001/56EC1E27-68E9-DD11-9AC4-003048678B5E.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-RECO/STARTUP_30X_v1/0001/6EDB7423-6EE9-DD11-9623-003048678C06.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-RECO/STARTUP_30X_v1/0001/840B687B-6EE9-DD11-91BA-001A92810AC8.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-RECO/STARTUP_30X_v1/0002/1ABE2464-1BEA-DD11-9E05-00304876A0ED.root'
  ),
  secondaryFileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/04144D27-73E9-DD11-84C2-0018F3D0968E.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/081EFC59-6FE9-DD11-95CD-001A92971BC8.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/225F6227-68E9-DD11-A902-003048678BC6.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/24154E3C-6EE9-DD11-9134-001A92971B28.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/243524DA-70E9-DD11-85CD-001A92811718.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/48F19758-67E9-DD11-9726-0030486792F0.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/4C5EF2AC-74E9-DD11-A678-001A92971ACC.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/58976E66-72E9-DD11-8C56-001731AF66C2.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/6E07805E-6FE9-DD11-9EFC-0017312B55A3.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/7ADBCF0C-6EE9-DD11-A016-0018F3D0961A.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/8AF4B11F-6EE9-DD11-BE97-003048679084.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/94429828-68E9-DD11-A4A1-003048678B34.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/A4810A48-6EE9-DD11-95AC-001BFCDBD1B6.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/BA95D827-73E9-DD11-85A2-001A9281170A.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/CE96A220-6EE9-DD11-BA2A-0030486790A0.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/D26C7B45-6EE9-DD11-8D35-001A92811722.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/D62144DA-70E9-DD11-BA30-001A92811710.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/DC508732-6EE9-DD11-85E9-00304875AB5D.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/DC5EE426-6EE9-DD11-84E1-0018F3D0969A.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/E02E2021-6EE9-DD11-B394-001A92971B82.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/E2E3C80C-6EE9-DD11-A0A3-0018F3D095F2.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/408558ED-76E9-DD11-8070-003048678B08.root',
       '/store/relval/CMSSW_3_0_0_pre7/RelValJpsiMM_Pt_20_inf/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/74447C27-1BEA-DD11-9577-0018F3D0968C.root'
  )
)
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(100)
)

process.load("HLTriggerOffline.HeavyFlavor.heavyFlavorValidationSequence_cff")
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.path = cms.Path(
  process.heavyFlavorValidationSequence
  *process.endOfProcess
)

process.outputmodule = cms.OutputModule("PoolOutputModule",
  outputCommands = cms.untracked.vstring(
    'drop *', 
    'keep *_MEtoEDMConverter_*_HEAVYFLAVORVALIDATION'
  ),
  fileName = cms.untracked.string('heavyFlavorValidation.root')
)
process.endpath = cms.EndPath(process.outputmodule)
