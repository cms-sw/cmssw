# PYTHON configuration file.
# Description:  Example of dijet mass plot
#               with corrected and uncorrected jets
#               for various jet algorithms
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
process.DijetMassCaloJets = cms.EDAnalyzer("DijetMassCaloJets",
    # Uncorrected CaloJets
    AKJets           = cms.string('ak5CaloJets'),
    ICJets           = cms.string('iterativeCone5CaloJets'),
    SCJets           = cms.string('sisCone5CaloJets'),
    KTJets           = cms.string('kt4CaloJets'),
    # Corrected CaloJets                                          
    AKCorrectedJets  = cms.string('L2L3CorJetAK5Calo'), 
    ICCorrectedJets  = cms.string('L2L3CorJetIC5Calo'), 
    SCCorrectedJets  = cms.string('L2L3CorJetSC5Calo'), 
    KTCorrectedJets  = cms.string('L2L3CorJetKT4Calo'),
    # Name of the output ROOT file containing the histograms 
    HistoFileName = cms.untracked.string('DijetMassCaloJets.root')
)

#############   User analyzer (PF jets) ##
process.DijetMassPFJets = cms.EDAnalyzer("DijetMassPFJets",
    # Uncorrected PFJets
    AKJets           = cms.string('ak5PFJets'),
    ICJets           = cms.string('iterativeCone5PFJets'),
    SCJets           = cms.string('sisCone5PFJets'),
    KTJets           = cms.string('kt4PFJets'),
    # Corrected PFJets                                          
    AKCorrectedJets  = cms.string('L2L3CorJetAK5PF'), 
    ICCorrectedJets  = cms.string('L2L3CorJetIC5PF'), 
    SCCorrectedJets  = cms.string('L2L3CorJetSC5PF'), 
    KTCorrectedJets  = cms.string('L2L3CorJetKT4PF'),
    # Name of the output ROOT file containing the histograms 
    HistoFileName = cms.untracked.string('DijetMassPFJets.root')
)


#############   User analyzer (gen jets) ##
# ak5GenJets are NOT there: First load the needed modules
process.load("RecoJets.Configuration.GenJetParticles_cff")
process.load("RecoJets.JetProducers.ak5GenJets_cfi")
process.DijetMassGenJets = cms.EDAnalyzer("DijetMassGenJets",
    # Uncorrected GenJets
    AKJets           = cms.string('ak5GenJets'),
    ICJets           = cms.string('iterativeCone5GenJets'),
    SCJets           = cms.string('sisCone5GenJets'),
    KTJets           = cms.string('kt4GenJets'),
    # Corrected GenJets  == Uncorrected GenJets                                        
    AKCorrectedJets  = cms.string('ak5GenJets'), 
    ICCorrectedJets  = cms.string('iterativeCone5GenJets'), 
    SCCorrectedJets  = cms.string('sisCone5GenJets'), 
    KTCorrectedJets  = cms.string('kt4GenJets'),
    # Name of the output ROOT file containing the histograms 
    HistoFileName = cms.untracked.string('DijetMassGenJets.root')
)


#############   Path       ###########################
process.p = cms.Path(process.L2L3CorJetAK5Calo * process.L2L3CorJetIC5Calo * process.L2L3CorJetSC5Calo * process.L2L3CorJetKT4Calo * process.DijetMassCaloJets)
process.p2 = cms.Path(process.L2L3CorJetAK5PF * process.L2L3CorJetIC5PF * process.L2L3CorJetSC5PF * process.L2L3CorJetKT4PF * process.DijetMassPFJets)
process.p3 = cms.Path(process.genParticlesForJets * process.ak5GenJets * process.DijetMassGenJets)

#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

