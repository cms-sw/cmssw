import FWCore.ParameterSet.Config as cms
from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos

ak4GenJetFlavourInfos = ak4JetFlavourInfos.clone(
                         jets = "ak4GenJetsNoNu",
                         bHadrons = cms.InputTag("selectedHadronsAndPartonsForGenJetsFlavourInfos","bHadrons"),
                         cHadrons = cms.InputTag("selectedHadronsAndPartonsForGenJetsFlavourInfos","cHadrons"),
                         partons  = cms.InputTag("selectedHadronsAndPartonsForGenJetsFlavourInfos","physicsPartons"),
                         leptons  = cms.InputTag("selectedHadronsAndPartonsForGenJetsFlavourInfos","leptons"),
)

