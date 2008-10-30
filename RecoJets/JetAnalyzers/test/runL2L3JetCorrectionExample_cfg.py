# PYTHON configuration file.
# Description:  Example of applying default (L2+L3) jet corrections.
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
#############   User analyzer (corrected jets) #######
process.corrected = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm  = cms.string('L2L3CorJetIC5Calo'),
    HistoFileName = cms.string('CorJetHisto.root'),
    NJets         = cms.int32(2)
)
#############   User analyzer (uncorrected jets) #####
process.uncorrected = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm    = cms.string('iterativeCone5CaloJets'),
    HistoFileName   = cms.string('CaloJetHisto.root'),
    NJets           = cms.int32(2)
)
#############   Path       ###########################
process.p = cms.Path(process.uncorrected * process.L2L3CorJetIC5Calo * process.corrected)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

