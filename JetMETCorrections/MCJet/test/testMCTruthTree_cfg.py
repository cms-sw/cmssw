import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
#############   Define the source file ###############
process.load("JetMETCorrections.MCJet.RelValQCD_cfi") 
#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_4_0_pre2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V10-v1/0003/D085615A-A5BD-DE11-8897-0026189437E8.root')
#)
#############    CaloJets ############################
process.caloMctruthTree = cms.EDAnalyzer("CaloMCTruthTreeProducer",
    jets               = cms.string('ak5CaloJets'),
    genjets            = cms.string('ak5GenJets'),
    histogramFile      = cms.string('ak5CaloMctruthTree.root')
)
#############    PFJets   ############################
process.pfMctruthTree = cms.EDAnalyzer("PFMCTruthTreeProducer",
    jets               = cms.string('ak5PFJets'),
    genjets            = cms.string('ak5GenJets'),
    histogramFile      = cms.string('ak5PFMctruthTree.root')
)
#############   Path       ###########################
process.p = cms.Path(process.caloMctruthTree * process.pfMctruthTree)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10
