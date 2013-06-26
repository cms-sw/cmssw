# PYTHON configuration file.
# Description:  Example of dijet ratio plot
#               with corrected and uncorrected jets
# Author: Kalanand Mishra
# Date:  22 - November - 2009 

import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
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

#############   User analyzer (calo jets) ##
process.DijetRatioCaloJets = cms.EDAnalyzer("DijetRatioCaloJets",
    # Uncorrected CaloJets
    UnCorrectedJets           = cms.string('ak5CaloJets'),
    # Corrected CaloJets                                          
    CorrectedJets  = cms.string('L2L3CorJetAK5Calo'), 
    # Name of the output ROOT file containing the histograms 
    HistoFileName = cms.untracked.string('DijetRatioCaloJets.root')
)

#############   User analyzer (PF jets) ##
process.DijetRatioPFJets = cms.EDAnalyzer("DijetRatioPFJets",
    # Uncorrected PFJets
    UnCorrectedJets          = cms.string('ak5PFJets'),
    # Corrected PFJets                                          
    CorrectedJets = cms.string('L2L3CorJetAK5PF'), 
    # Name of the output ROOT file containing the histograms 
    HistoFileName = cms.untracked.string('DijetRatioPFJets.root')
)


#############   User analyzer (gen jets) ##
# ak5GenJets are NOT there: First load the needed modules
process.load("RecoJets.Configuration.GenJetParticles_cff")
process.load("RecoJets.JetProducers.ak5GenJets_cfi")
process.DijetRatioGenJets = cms.EDAnalyzer("DijetRatioGenJets",
    # Uncorrected GenJets
    UnCorrectedJets          = cms.string('ak5GenJets'),
    # Corrected GenJets  == Uncorrected GenJets   
    CorrectedJets  = cms.string('ak5GenJets'), 
    # Name of the output ROOT file containing the histograms 
    HistoFileName = cms.untracked.string('DijetRatioGenJets.root')
)


#############   Path       ###########################
process.p = cms.Path(process.L2L3CorJetAK5Calo * process.DijetRatioCaloJets)
process.p2 = cms.Path(process.L2L3CorJetAK5PF * process.DijetRatioPFJets)
process.p3 = cms.Path(process.genParticlesForJets *
                         process.ak5GenJets * process.DijetRatioGenJets)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

