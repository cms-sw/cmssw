import FWCore.ParameterSet.Config as cms

process = cms.Process("ExoticaDQM")
process.load("DQM.Physics.ExoticaDQM_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.Components.DQMFileSaver_cfi")
#process.DQM.collectorHost = ''
process.dqmSaver.workflow = cms.untracked.string('/Physics/ExoticDQM/DataSet')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(4000)
)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
# Works on GEN-SIM-RECO or AODSIM, but not MINIAOD format.
readFiles.extend( [
       'root://xrootd.unl.edu//store/relval/CMSSW_7_6_0_pre1/RelValDisplacedSUSY_stopToBottom_M_300_1000mm_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v1-v1/00000/66DB5704-1430-E511-AC0F-002618943865.root',
       'root://xrootd.unl.edu//store/relval/CMSSW_7_6_0_pre1/RelValDisplacedSUSY_stopToBottom_M_300_1000mm_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v1-v1/00000/B083D0CC-1230-E511-AB99-0025905A610A.root',
       'root://xrootd.unl.edu//store/relval/CMSSW_7_6_0_pre1/RelValDisplacedSUSY_stopToBottom_M_300_1000mm_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v1-v1/00000/EADDAE49-1430-E511-90BA-0026189438D9.root' ] );

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
#=== Next 5 lines needed for jet DQM plots
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')
process.load('JetMETCorrections.Configuration.JetCorrectorsForReco_cff') 
process.load('JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff')
process.prefer("ak4CaloL2L3")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('ExoticaDQM'   )
process.MessageLogger.cerr.ExoticaDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))

## check the event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## this is a full sequence to test the functionality of all modules
process.p = cms.Path(
    process.jetCorrectorsForReco +
    process.ExoticaDQM +      
    ## save histograms
    process.dqmSaver
)

## Options and Output Report
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
