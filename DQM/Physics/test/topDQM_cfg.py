import FWCore.ParameterSet.Config as cms

process = cms.Process("TopDQM")
process.load("DQM.Physics.topSingleLeptonDQM_cfi")
process.load("DQM.Physics.topDiLeptonOfflineDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''
process.dqmSaver.workflow = cms.untracked.string('/Physics/TopSingleLeptonDQM/DataSet')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource"
    ,fileNames = cms.untracked.vstring(
      '/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-RECO/MC_38Y_V12-v1/0039/2ABE8934-E7D1-DF11-9985-003048678C9A.root'
     #'/store/relval/CMSSW_3_8_4/RelValTTbar/GEN-SIM-RECO/START38_V12-v1/0024/1A650A81-83C2-DF11-B355-002618FDA287.root'
    )
)

## apply VBTF electronID (needed for the current implementation
## of topSingleElectronDQMLoose and topSingleElectronDQMMedium)
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("DQM.Physics.topElectronID_cff")

## load jet corrections
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
## --------------------------------------------------------------------
## Frontier Conditions: (adjust accordingly!!!)
##
## For CMSSW_3_8_X MC use             ---> 'START38_V12::All'
## For Data (38X re-processing) use   ---> 'GR_R_38X_V13::All'
##
## For more details have a look at: WGuideFrontierConditions
## --------------------------------------------------------------------
process.GlobalTag.globaltag = 'START38_V12::All' 
process.load('JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff')
process.prefer("ak4CaloL2L3")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TopSingleLeptonDQM'   )
process.MessageLogger.cerr.TopSingleLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.categories.append('TopDiLeptonOfflineDQM')
process.MessageLogger.cerr.TopDiLeptonOfflineDQM = cms.untracked.PSet(limit = cms.untracked.int32(1))

## check the event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## this is a full sequence to test the functionality of all modules
process.p = cms.Path(
    #process.content *
    ## common dilepton monitoring
    process.simpleEleId70cIso          *
    process.topDiLeptonOfflineDQM      +
    ## common lepton plus jets monitoring
    process.topSingleLeptonDQM         +
    ## muon plus jets monitoring
    process.topSingleMuonLooseDQM      +    
    process.topSingleMuonMediumDQM     +
    ## electron plus jets monitoring
    process.topSingleElectronLooseDQM  +    
    process.topSingleElectronMediumDQM +
    ## save histograms
    process.dqmSaver
)

## Options and Output Report
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
