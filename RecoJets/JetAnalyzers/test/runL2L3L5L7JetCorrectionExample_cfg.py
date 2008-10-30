# PYTHON configuration file.
# Description:  Example of applying L2+L3+L5+L7 jet corrections.
# Author: K. Kousouris
# Date:  25 - August - 2008
import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/Summer08/QCDDiJetPt80to120/GEN-SIM-RECO/IDEAL_V9_v1/0000/009AC3E3-BF97-DD11-93B5-00093D13BB43.root')
)
#############   Define the L2 correction service #####
process.L2RelativeJetCorrector = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('Summer08_L2Relative_IC5Calo'),
    label = cms.string('L2RelativeJetCorrector')
)
#############   Define the L3 correction service #####
process.L3AbsoluteJetCorrector = cms.ESSource("L3AbsoluteCorrectionService", 
    tagName = cms.string('Summer08_L3Absolute_IC5Calo'),
    label = cms.string('L3AbsoluteJetCorrector')
)
#############   Define the L5 correction service #####
process.L5JetCorrector = cms.ESSource("L5FlavorCorrectionService",
    section = cms.string('uds'),  
    tagName = cms.string('L5Flavor_fromQCD_iterativeCone5'),
    label = cms.string('L5FlavorJetCorrector')
)
#############   Define the L7 correction service #####
process.L7JetCorrector = cms.ESSource("L7PartonCorrectionService", 
    section = cms.string('qJ'),
    tagName = cms.string('L7parton_IC5_080301'),
    label = cms.string('L7PartonJetCorrector')
)
#############   Define the chain corrector service ###
process.L2L3L5L7JetCorrector = cms.ESSource("JetCorrectionServiceChain",  
    correctors = cms.vstring('L2RelativeJetCorrector','L3AbsoluteJetCorrector','L5FlavorJetCorrector','L7PartonJetCorrector'),
    label = cms.string('L2L3L5L7JetCorrector')
)
#############   Define the chain corrector module ####
process.L2L3L5L7CorJet = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('L2L3L5L7JetCorrector')
)
# set the record's IOV. Must be defined once. Choose ANY correction service. #
process.prefer("L2L3L5L7JetCorrector") 
#############   Plots of corrected Jet collection ####
process.plots = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm  = cms.string('L2L3L5L7CorJet'),
    HistoFileName = cms.string('L2L3L5L7CaloJetCorExample.root'),
    NJets         = cms.int32(100)
)
#############   Path       ###########################
process.p = cms.Path(process.L2L3L5L7CorJet * process.plots)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10
