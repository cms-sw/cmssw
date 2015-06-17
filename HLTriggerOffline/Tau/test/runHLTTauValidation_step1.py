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
        '/store/hidata/HIRun2011/HIMinBiasUPC/RAW/v1/000/181/913/06B6DB7E-C810-E111-AE7C-001D09F2305C.root'
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




                                  

