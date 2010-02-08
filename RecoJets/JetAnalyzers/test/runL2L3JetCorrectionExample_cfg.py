# PYTHON configuration file.
# Description:  Example of applying default (L2+L3) jet corrections.
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
#############   User analyzer (corrected calo jets) ##
process.correctedSC5Calo = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm  = cms.string('L2L3CorJetSC5Calo'),
    HistoFileName = cms.string('CorJetHisto_SC5Calo.root'),
    NJets         = cms.int32(2)
)
#############   User analyzer (corrected pf jets) ##
process.correctedSC5PF = cms.EDAnalyzer("PFJetPlotsExample",
    JetAlgorithm  = cms.string('L2L3CorJetSC5PF'),
    HistoFileName = cms.string('CorJetHisto_SC5PF.root'),
    NJets         = cms.int32(2)
)
#############   User analyzer (uncorrected jets) #####
process.uncorrected = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm    = cms.string('sisCone5CaloJets'),
    HistoFileName   = cms.string('CaloJetHisto_SC5Calo.root'),
    NJets           = cms.int32(2)
)
#############   Path       ###########################
process.p = cms.Path(process.uncorrected * process.L2L3CorJetSC5Calo * process.L2L3CorJetSC5PF * process.correctedSC5Calo * process.correctedSC5PF)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

