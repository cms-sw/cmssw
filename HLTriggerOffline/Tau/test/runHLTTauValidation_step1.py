import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")


process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/scratch/bachtis/RECO-ZTT-GEN-SIM-DIGI-0013.root')
)

process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

# Conditions: fake or frontier
# process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'STARTUP_V4::All'

process.load("Configuration.StandardSequences.L1Emulator_cff")
# Choose a menu/prescale/mask from one of the choices
# in L1TriggerConfig.L1GtConfigProducers.Luminosity
process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")

# run HLT
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("HLTrigger.Configuration.HLT_2E30_cff")
process.schedule = process.HLTSchedule

process.hltL1gtTrigReport = cms.EDAnalyzer( "L1GtTrigReport",
    UseL1GlobalTriggerRecord = cms.bool( False ),
    L1GtRecordInputTag = cms.InputTag( "hltGtDigis" )
)
process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)
process.HLTAnalyzerEndpath = cms.EndPath( process.hltL1gtTrigReport + process.hltTrigReport )
process.schedule.append(process.HLTAnalyzerEndpath)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.DQMStore = cms.Service("DQMStore")

#Load DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#Load the Validation
process.load("HLTriggerOffline.Tau.Validation.HLTTauValidation_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cff")

#Load The Post processor
process.load("HLTriggerOffline.Tau.Validation.HLTTauPostValidation_cfi")


process.output = cms.OutputModule("PoolOutputModule",
           outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_*'),

           fileName = cms.untracked.string('test.root')
                                  )
#Define the Paths
process.validation = cms.Path(process.HLTTauVal)
process.postProcess = cms.EndPath(process.MEtoEDMConverter+process.output)
process.schedule.append(process.validation)
process.schedule.append(process.postProcess)



                                  

