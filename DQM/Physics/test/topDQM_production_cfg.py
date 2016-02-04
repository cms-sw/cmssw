import FWCore.ParameterSet.Config as cms

process = cms.Process('TOPDQM')

## imports of standard configurations
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## --------------------------------------------------------------------
## Frontier Conditions: (adjust accordingly!!!)
##
## For CMSSW_3_8_X MC use             ---> 'START38_V12::All'
## For Data (38X re-processing) use   ---> 'GR_R_38X_V13::All'
## For Data (38X prompt reco) use     ---> 'GR10_P_V10::All'
##
## For more details have a look at: WGuideFrontierConditions
## --------------------------------------------------------------------
##process.GlobalTag.globaltag = 'START38_V12::All'
##process.GlobalTag.globaltag = 'GR_R_38X_V13::All' 
process.GlobalTag.globaltag   = 'GR10_P_V10::All'

## input file(s) for testing
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
     '/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-RECO/MC_38Y_V12-v1/0039/2ABE8934-E7D1-DF11-9985-003048678C9A.root'
    #,'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-RECO/MC_38Y_V12-v1/0039/740F4118-E3D1-DF11-90EF-001A92971B32.root'
    #,'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-RECO/MC_38Y_V12-v1/0039/80C7AB1A-E4D1-DF11-AFF2-00261894388D.root'
    #,'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-RECO/MC_38Y_V12-v1/0039/829A469A-E5D1-DF11-B64A-001A92810AC0.root'
    #,'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-RECO/MC_38Y_V12-v1/0039/D06EF8B3-E3D1-DF11-9F56-001A92971BA0.root'
    #,'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-RECO/MC_38Y_V12-v1/0039/ECB3B1FA-E1D1-DF11-87E5-003048678AE2.root'
    #'/store/relval/CMSSW_3_8_4/RelValTTbar/GEN-SIM-RECO/START38_V12-v1/0024/1A650A81-83C2-DF11-B355-002618FDA287.root'
    )
)

## number of events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

## apply VBTF electronID (needed for the current implementation
## of topSingleElectronDQMLoose and topSingleElectronDQMMedium)
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("DQM.Physics.topElectronID_cff")

## output
process.output = cms.OutputModule("PoolOutputModule",
  fileName       = cms.untracked.string('topDQM_production.root'),
  outputCommands = cms.untracked.vstring(
    'drop *_*_*_*',
    'keep *_*_*_TOPDQM',
    'drop *_TriggerResults_*_TOPDQM',
    'drop *_simpleEleId70cIso_*_TOPDQM',
  ),
  splitLevel     = cms.untracked.int32(0),
  dataset = cms.untracked.PSet(
    dataTier   = cms.untracked.string(''),
    filterName = cms.untracked.string('')
  )
)

## load jet corrections
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
process.prefer("ak5CaloL2L3")

## check the event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TopSingleLeptonDQM'   )
process.MessageLogger.cerr.TopSingleLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.categories.append('TopDiLeptonOfflineDQM')
process.MessageLogger.cerr.TopDiLeptonOfflineDQM = cms.untracked.PSet(limit = cms.untracked.int32(1))


## path definitions
process.p      = cms.Path(
   #process.content *
    process.simpleEleId70cIso          *
    process.topDiLeptonOfflineDQM      +
    process.topSingleLeptonDQM         +
    process.topSingleMuonLooseDQM      +    
    process.topSingleMuonMediumDQM     +
    process.topSingleElectronLooseDQM  +    
    process.topSingleElectronMediumDQM
)
process.endjob = cms.Path(
    process.endOfProcess
)
process.fanout = cms.EndPath(
    process.output
)

## schedule definition
process.schedule = cms.Schedule(
    process.p,
    process.endjob,
    process.fanout
)
