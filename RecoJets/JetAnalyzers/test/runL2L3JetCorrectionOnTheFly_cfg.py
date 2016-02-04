# PYTHON configuration file for class: JetCorExample
# Description:  Example of simple EDAnalyzer for correcting jets on the fly.
# Author: K. Kousouris
# Date:  02 - September - 2009
import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V3-v1/0007/9E83A122-E978-DE11-9D04-001D09F23C73.root')
)
#############   Include the jet corrections ##########
process.load("JetMETCorrections.Configuration.L2L3Corrections_Summer09_cff")
# set the record's IOV. Must be defined once. Choose ANY correction service. #
process.prefer("L2L3JetCorrectorSC5Calo") 
#############   Correct Calo Jets on the fly #########
process.calo = cms.EDAnalyzer("CaloJetCorExample",
    JetAlgorithm         = cms.string('sisCone5CaloJets'),
    HistoFileName        = cms.string('CaloJetCorOnTheFlyExample_SC5Calo.root'),
    JetCorrectionService = cms.string('L2L3JetCorrectorSC5Calo')
)
#############   Correct PF Jets on the fly #########
process.calo = cms.EDAnalyzer("PFJetCorExample",
    JetAlgorithm         = cms.string('sisCone5PFJets'),
    HistoFileName        = cms.string('PFJetCorOnTheFlyExample_SC5PF.root'),
    JetCorrectionService = cms.string('L2L3JetCorrectorSC5PF')
)
#############   Path       ###########################
process.p = cms.Path(process.calo)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

