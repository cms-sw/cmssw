import FWCore.ParameterSet.Config as cms

process = cms.Process('TOPDQM')

## imports of standard configurations
#process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.Services_cff')

## --------------------------------------------------------------------
## Frontier Conditions: (adjust accordingly!!!)
##
## For CMSSW_3_8_X MC use             ---> 'START38_V12::All'
## For Data (38X re-processing) use   ---> 'GR_R_38X_V13::All'
## For Data (38X prompt reco) use     ---> 'GR10_P_V10::All'
##
## For more details have a look at: WGuideFrontierConditions
## --------------------------------------------------------------------
##process.GlobalTag.globaltag = 'GR_R_42_V14::All' 
#process.GlobalTag.globaltag = 'MCRUN2_74_V9'
#process.GlobalTag.globaltag = 'auto:startup_GRun'
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:100X_mc2017_realistic', '')
#dbs search --query 'find file where site=srm-eoscms.cern.ch and dataset=/RelValTTbar/CMSSW_7_0_0_pre3-PRE_ST62_V8-v1/GEN-SIM-RECO'
#dbs search --query 'find dataset where dataset=/RelValTTbar/CMSSW_7_0_0_pre6*/GEN-SIM-RECO'

## input file(s) for testing
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring("file:input.root',")
    fileNames = cms.untracked.vstring(
	"/store/relval/CMSSW_10_0_0_pre2/RelValTTbar_13/GEN-SIM-RECO/100X_mc2017_realistic_v1-v1/20000/1CD8D6F0-AFDC-E711-B2BC-0CC47A78A478.root"

#    "/store/relval/CMSSW_9_4_0_pre2/RelValTTbar_13/GEN-SIM-RECO/94X_upgrade2018_realistic_v1-v1/10000/5CD4DB82-83A9-E711-B067-0025905B8560.root",
#    "/store/relval/CMSSW_9_4_0_pre2/RelValTTbar_13/GEN-SIM-RECO/94X_upgrade2018_realistic_v1-v1/10000/74FB294F-7FA9-E711-9582-0CC47A4D7626.root",
#    "/store/relval/CMSSW_9_4_0_pre2/RelValTTbar_13/GEN-SIM-RECO/94X_upgrade2018_realistic_v1-v1/10000/78446D4B-7FA9-E711-A556-0CC47A78A42C.root",
    #"/store/relval/CMSSW_6_2_0_pre1-START61_V8/RelValTTbarLepton/GEN-SIM-RECO/v1/00000/C6CC53CC-6E6D-E211-8EAB-003048D3756A.root',"
    
    #/RelValTTbar/CMSSW_7_0_0_pre6-PRE_ST62_V8-v1/GEN-SIM-RECO
    #'/store/relval/CMSSW_7_0_0_pre6/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/B627D32C-0B3C-E311-BBE6-0026189438E6.root',
    #'/store/relval/CMSSW_7_0_0_pre6/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/72477A84-F93B-E311-BF63-003048FFD720.root',
    #'/store/relval/CMSSW_7_0_0_pre6/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/12A06D7A-F93B-E311-AA64-003048678BEA.root'
        #    '/store/relval/CMSSW_7_1_0_pre4/RelValTTbarLepton_13/GEN-SIM-RECO/POSTLS171_V1-v2/00000/48ED95A2-66AA-E311-9865-02163E00E5AE.root'
    #'/store/relval/CMSSW_8_0_0_pre5/RelValTTbarLepton_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v1-v1/00000/120469B3-20C5-E511-A321-0026189438CB.root'
#'/store/relval/CMSSW_7_6_0_pre1/RelValTTbarLepton_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v1-v1/00000/303083D5-F82F-E511-8B50-0025905B8576.root'
    )
)

## number of events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

## apply VBTF electronID (needed for the current implementation
## of topSingleElectronDQMLoose and topSingleElectronDQMMedium)
#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("DQM.Physics.topElectronID_cff")
process.load('Configuration/StandardSequences/Reconstruction_cff')


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
process.load("JetMETCorrections.Configuration.JetCorrectors_cff")

## check the event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TopSingleLeptonDQM'   )
process.MessageLogger.cerr.TopSingleLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.categories.append('TopDiLeptonOfflineDQM')
process.MessageLogger.cerr.TopDiLeptonOfflineDQM = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.categories.append('SingleTopTChannelLeptonDQM'   )
process.MessageLogger.cerr.SingleTopTChannelLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.cerr.FwkReport.reportEvery = 100


process.load("DQM.Physics.topSingleLeptonDQM_cfi")
process.load("DQM.Physics.singleTopDQM_cfi")


## path definitions
process.p      = cms.Path(
#    process.simpleEleId70cIso          *
    #process.DiMuonDQM                  +
    #process.DiElectronDQM              +
    #process.ElecMuonDQM                +
    #process.topSingleMuonLooseDQM      +
    process.dqmAk4PFCHSL1FastL2L3CorrectorChain * process.topSingleMuonMediumDQM     +
    #process.topSingleElectronLooseDQM  +
    process.dqmAk4PFCHSL1FastL2L3CorrectorChain * process.topSingleElectronMediumDQM +
    process.dqmAk4PFCHSL1FastL2L3CorrectorChain * process.singleTopMuonMediumDQM      +
    process.dqmAk4PFCHSL1FastL2L3CorrectorChain * process.singleTopElectronMediumDQM
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
