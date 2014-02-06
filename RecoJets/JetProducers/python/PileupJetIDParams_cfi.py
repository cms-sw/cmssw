import FWCore.ParameterSet.Config as cms
from RecoJets.JetProducers.PileupJetIDCutParams_cfi import *

####################################################################################################################  
full_5x_chs = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("RecoJets/JetProducers/data/TMVAClassification_5x_BDT_chsFullPlusRMS.weights.xml.gz"),
 tmvaMethod  = cms.string("BDT_chsFullPlusRMS"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "frac01",
    "frac02",
    "frac03",
    "frac04",
    "frac05",
    "dR2Mean",
    "nvtx",
    "nNeutrals",
    "beta",
    "betaStar",
    "dZ",
    "nCharged",
    ),
 tmvaSpectators = cms.vstring(
    "jetPt",
    "jetEta",
    ),
 JetIdParams = full_5x_chs_wp,
 label = cms.string("full")
 )
####################################################################################################################  
cutbased = cms.PSet( 
 impactParTkThreshold = cms.double(1.),
 cutBased = cms.bool(True),
 JetIdParams = PuJetIdCutBased_wp,
 label = cms.string("cutbased")
 )

