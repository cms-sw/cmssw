# PYTHON configuration file.
# Description:  Example of applying L2+L3+L4 jet corrections.
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
#############   Define the L2 correction service #####
process.L2RelativeJetCorrector = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('Summer09_L2Relative_SC5Calo'),
    label = cms.string('L2RelativeJetCorrector')
)
#############   Define the L3 correction service #####
process.L3AbsoluteJetCorrector = cms.ESSource("L3AbsoluteCorrectionService", 
    tagName = cms.string('Summer09_L3Absolute_SC5Calo'),
    label = cms.string('L3AbsoluteJetCorrector')
)
#############   Define the EMF correction service ####
process.L4JetCorrector = cms.ESSource("L4EMFCorrectionService", 
    tagName = cms.string('L4EMF_IC5'), # IMPORTANT: the L4 correction was derived from IC5 but it is the same for all algos
    label = cms.string('L4EMFJetCorrector')
)
#############   Define the chain corrector service ###
process.L2L3L4JetCorrector = cms.ESSource("JetCorrectionServiceChain",  
    correctors = cms.vstring('L2RelativeJetCorrector','L3AbsoluteJetCorrector','L4EMFJetCorrector'),
    label = cms.string('L2L3L4JetCorrector')
)
#############   Define the chain corrector module ####
process.L2L3L4CorJet = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("sisCone5CaloJets"),
    correctors = cms.vstring('L2L3L4JetCorrector')
)
# set the record's IOV. Must be defined once. Choose ANY correction service. #
process.prefer("L2L3L4JetCorrector") 
#############   Plots of corrected Jet collection ####
process.plots = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm  = cms.string('L2L3L4CorJet'),
    HistoFileName = cms.string('L2L3L4CaloJetCorExample.root'),
    NJets         = cms.int32(100)
)
#############   Path       ###########################
process.p = cms.Path(process.L2L3L4CorJet * process.plots)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10
