import FWCore.ParameterSet.Config as cms
process = cms.Process("SKIM")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('JetMETCorrections.Utilities.TriggerFilter_cff')
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_1_0_pre11/RelValQCD_Pt_120_170/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/ECE80EA3-DE64-DE11-B97C-00304875AA77.root')
)

#############   output module ########################
process.compress = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_sisCone5CaloJets_*_*',
        'keep *_sisCone7CaloJets_*_*', 
        'keep *_towerMaker_*_*',
        'keep *_met_*_*'),
    fileName = cms.untracked.string('JetAOD.root')
)
#############   Path       ###########################
process.p = cms.Path(process.skimTriggerBit)
process.outpath = cms.EndPath(process.compress)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

