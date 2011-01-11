# PYTHON configuration file for class: JetPlotsExample
# Description:  Example of simple EDAnalyzer for jets.
# Author: K. Kousouris
# Date:  25 - August - 2008
# Modified: Kalanand Mishra
# Date:  11 - January - 2011 (for CMS Data Analysis School jet exercise)


import FWCore.ParameterSet.Config as cms

isMC = True
NJetsToKeep = 2

process = cms.Process("Ana")
#############   Format MessageLogger #################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/mc/Fall10/QCD_Pt_80to120_TuneZ2_7TeV_pythia6/GEN-SIM-RECO/START38_V12-v1/0000/FEF4D100-4CCB-DF11-94CB-00E08178C12F.root')
)
process.source.inputCommands = cms.untracked.vstring("keep *","drop *_MEtoEDMConverter_*_*")
#############   Calo Jets  ###########################
process.calo = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm  = cms.string('ak5CaloJets'),
    HistoFileName = cms.string('CaloJetPlotsExample.root'),
    NJets         = cms.int32(NJetsToKeep)
)
#############   PF Jets    ###########################
process.pf = cms.EDAnalyzer("PFJetPlotsExample",
    JetAlgorithm  = cms.string('ak5PFJets'),
    HistoFileName = cms.string('PFJetPlotsExample.root'),
    NJets         = cms.int32(NJetsToKeep)
)
#############   JPT Jets    ###########################
process.jpt = cms.EDAnalyzer("JPTJetPlotsExample",
    JetAlgorithm  = cms.string('JetPlusTrackZSPCorJetAntiKt5'),
    HistoFileName = cms.string('JPTJetPlotsExample.root'),
    NJets         = cms.int32(NJetsToKeep)
)
#############   Gen Jets   ###########################
if isMC:
    process.gen = cms.EDAnalyzer("GenJetPlotsExample",
        JetAlgorithm  = cms.string('ak5GenJets'),
        HistoFileName = cms.string('GenJetPlotsExample.root'),
        NJets         = cms.int32(NJetsToKeep)
    )

#############   Path       ###########################
if isMC:    
    process.p = cms.Path(process.calo*process.pf*process.jpt*process.gen)
else:
    process.p = cms.Path(process.calo*process.pf*process.jpt)    


