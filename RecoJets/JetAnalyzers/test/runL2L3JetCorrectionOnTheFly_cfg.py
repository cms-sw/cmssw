# PYTHON configuration file for class: JetCorExample
# Description:  Example of simple EDAnalyzer for correcting jets on the fly.
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
#############   Include the jet corrections ##########
process.load("JetMETCorrections.Configuration.L2L3Corrections_Summer08_cff")
# set the record's IOV. Must be defined once. Choose ANY correction service. #
process.prefer("L2L3JetCorrectorIC5Calo") 
#############   Correct Calo Jets on the fly #########
process.calo = cms.EDAnalyzer("CaloJetCorExample",
    JetAlgorithm         = cms.string('iterativeCone5CaloJets'),
    HistoFileName        = cms.string('CaloJetCorOnTheFlyExample.root'),
    JetCorrectionService = cms.string('L2L3JetCorrectorIC5Calo')
)
#############   Path       ###########################
process.p = cms.Path(process.calo)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

