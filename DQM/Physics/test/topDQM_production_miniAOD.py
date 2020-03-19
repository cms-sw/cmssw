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
##process.GlobalTag.globaltag = 'GR_R_42_V14::All' 
#process.GlobalTag.globaltag = 'MCRUN2_74_V9'

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')
 


#dbs search --query 'find file where site=srm-eoscms.cern.ch and dataset=/RelValTTbar/CMSSW_7_0_0_pre3-PRE_ST62_V8-v1/GEN-SIM-RECO'
#dbs search --query 'find dataset where dataset=/RelValTTbar/CMSSW_7_0_0_pre6*/GEN-SIM-RECO'

## input file(s) for testing
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring("file:input.root',")
    fileNames = cms.untracked.vstring(

	'/store/relval/CMSSW_10_0_0_pre2/RelValTTbar_13/MINIAODSIM/100X_mc2017_realistic_v1-v1/20000/4E3E8C73-C1DC-E711-9DAC-0CC47A7C340E.root'
#    	'/store/relval/CMSSW_9_2_4/RelValTTbarLepton_13/MINIAODSIM/92X_upgrade2017_realistic_v2-v1/00000/A4DA3BEB-3C5C-E711-AA32-003048FFD79E.root',
#'/store/relval/CMSSW_9_2_4/RelValTTbarLepton_13/MINIAODSIM/92X_upgrade2017_realistic_v2-v1/00000/D83C1AEF-3C5C-E711-82F4-0CC47A4D7600.root'

    

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
#process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
#process.prefer("ak4PFL2L3")

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


process.load("DQM.Physics.topSingleLeptonDQM_miniAOD_cfi")
process.load("DQM.Physics.singleTopDQM_miniAOD_cfi")


## path definitions
process.p      = cms.Path(
#    process.simpleEleId70cIso          *
#    process.DiMuonDQM                  +
#    process.DiElectronDQM              +
#    process.ElecMuonDQM                +
    #process.topSingleMuonLooseDQM      +
    process.topSingleMuonMediumDQM_miniAOD  +
    #process.topSingleElectronLooseDQM  +
    process.topSingleElectronMediumDQM_miniAOD +
    process.singleTopMuonMediumDQM_miniAOD     +
    process.singleTopElectronMediumDQM_miniAOD
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
