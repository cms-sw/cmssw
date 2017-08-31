import FWCore.ParameterSet.Config as cms
from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos

ak4GenJetFlavourInfos = ak4JetFlavourInfos.clone(jets = "ak4GenJetsNoNu")

