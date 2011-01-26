# PYTHON configuration file.
# Description:  Example of applying default (L2+L3) jet corrections.
# Author: K. Kousouris
# Date:  02 - September - 2009
# Date:  22 - November - 2009: Kalanand Mishra: Modified for 3.3.X (re-Reco) corrections

import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/mc/Summer09/QCDFlat_Pt15to3000/GEN-SIM-RECO/MC_31X_V9_7TeV-v1/0000/FABD2A94-C0D3-DE11-B6FD-00237DA13C2E.root')
)
process.source.inputCommands = cms.untracked.vstring("keep *","drop *_MEtoEDMConverter_*_*")

#############   Include the jet corrections ##########
process.load("JetMETCorrections.Configuration.L2L3Corrections_Summer09_7TeV_ReReco332_cff")
# set the record's IOV. Must be defined once. Choose ANY correction service. #
process.prefer("L2L3JetCorrectorAK5Calo") 
#############   User analyzer (corrected calo jets) ##
process.correctedAK5Calo = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm  = cms.string('L2L3CorJetAK5Calo'),
    HistoFileName = cms.string('CorJetHisto_AK5Calo.root'),
    NJets         = cms.int32(2)
)
#############   User analyzer (corrected pf jets) ##
process.correctedAK5PF = cms.EDAnalyzer("PFJetPlotsExample",
    JetAlgorithm  = cms.string('L2L3CorJetAK5PF'),
    HistoFileName = cms.string('CorJetHisto_AK5PF.root'),
    NJets         = cms.int32(2)
)
#############   User analyzer (uncorrected jets) #####
process.uncorrected = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm    = cms.string('ak5CaloJets'),
    HistoFileName   = cms.string('CaloJetHisto_AK5Calo.root'),
    NJets           = cms.int32(2)
)
#############   Path       ###########################
process.p = cms.Path(process.uncorrected * process.L2L3CorJetAK5Calo * process.L2L3CorJetAK5PF * process.correctedAK5Calo * process.correctedAK5PF)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

