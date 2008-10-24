import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
#############   Define the source file ###############
#process.load("JetMETCorrections.MCJet.QCDDiJet_50_80_cfi") 
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/Summer08/QCDDiJetPt50to80/GEN-SIM-RECO/IDEAL_V9_v3/0001/0079B347-9E9A-DD11-ADD9-003048C17FC0.root')
)
#############    CaloJets ############################
process.caloMctruthTree = cms.EDAnalyzer("CaloMCTruthTreeProducer",
    jets               = cms.string('iterativeCone5CaloJets'),
    genjets            = cms.string('iterativeCone5GenJets'),
    histogramFile      = cms.string('ic05CaloMctruthTree.root')
)
#############    PFJets   ############################
process.pfMctruthTree = cms.EDAnalyzer("PFMCTruthTreeProducer",
    jets               = cms.string('iterativeCone5PFJets'),
    genjets            = cms.string('iterativeCone5GenJets'),
    histogramFile      = cms.string('ic05PFMctruthTree.root')
)
#############   Path       ###########################
process.p = cms.Path(process.caloMctruthTree * process.pfMctruthTree)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10
