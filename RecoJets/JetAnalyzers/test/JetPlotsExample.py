# PYTHON configuration file for class: JetPlotsExample
# Description:  Example of simple EDAnalyzer for jets.
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
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RECO/STARTUP31X_V8_FastSim_Early10TeVCollision-v1/0001/0E3E1B64-56B7-DE11-A8E1-00304867924E.root')
)
#############   Calo Jets  ###########################
process.calo = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm  = cms.string('iterativeCone5CaloJets'),
    HistoFileName = cms.string('CaloJetPlotsExample.root'),
    NJets         = cms.int32(2)
)
#############   Gen Jets   ###########################
process.gen = cms.EDAnalyzer("GenJetPlotsExample",
    JetAlgorithm  = cms.string('iterativeCone5GenJets'),
    HistoFileName = cms.string('GenJetPlotsExample.root'),
    NJets         = cms.int32(2)
)
#############   PF Jets    ###########################
process.pf = cms.EDAnalyzer("PFJetPlotsExample",
    JetAlgorithm  = cms.string('iterativeCone5PFJets'),
    HistoFileName = cms.string('PFJetPlotsExample.root'),
    NJets         = cms.int32(2)
)
#############   Path       ###########################
process.p = cms.Path(process.calo*process.gen*process.pf)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

