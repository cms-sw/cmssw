import FWCore.ParameterSet.Config as cms
process = cms.Process("SKIM")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('JetMETCorrections.Configuration.jecHLTFilters_cfi')
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_1_0_pre11/RelValQCD_Pt_120_170/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/ECE80EA3-DE64-DE11-B97C-00304875AA77.root')
)

#############   Path       ###########################
process.skimPath = cms.Path(process.HLTPhotons)

#############   output module ########################
process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('skimPath')), 
    fileName = cms.untracked.string('SkimPhotons.root')
)

process.p = cms.EndPath(process.out)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

