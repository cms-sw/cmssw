import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTVAL")


process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/00C9C666-B3B2-DD11-A3E5-000423D9853C.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/0AAE2E10-B3B2-DD11-AC65-001617E30CC8.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/0C25E503-B3B2-DD11-8BF3-000423D98EA8.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/1469C60D-B3B2-DD11-BF43-000423D99F3E.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/16FCDE64-B3B2-DD11-B009-000423D6B42C.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/223CA255-B3B2-DD11-A293-000423D6C8E6.root',
                                      )
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.DQMStore = cms.Service("DQMStore")

#Load DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#Load the Validation
process.load("HLTriggerOffline.Tau.Validation.HLTTauValidation_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cff")


process.output = cms.OutputModule("PoolOutputModule",
           outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_*'),

           fileName = cms.untracked.string('test.root')
                                  )
#Define the Paths
process.validation = cms.Path(process.HLTTauVal)
process.postProcess = cms.EndPath(process.MEtoEDMConverter+process.output)
process.schedule=cms.Schedule(process.validation,process.postProcess)




                                  

