import FWCore.ParameterSet.Config as cms
from RecoJets.JetProducers.JetIdParams_cfi import *

####################################################################################################################  
full_53x = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("CondFormats/JetMETObjects/data/TMVAClassificationCategory_JetID_53X_Dec2012.weights.xml"),
 tmvaMethod  = cms.string("JetIDMVAHighPt"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "nvtx"     ,
    "dZ"       , 
    "beta"     , 
    "betaStar" , 
    "nCharged" , 
    "nNeutrals", 
    "dR2Mean"  , 
    "ptD"      , 
    "frac01"   , 
    "frac02"   , 
    "frac03"   , 
    "frac04"   , 
    "frac05"   , 
    ),
 tmvaSpectators = cms.vstring(
    "jetPt",
    "jetEta",
    "jetPhi"
    ),
 JetIdParams = full_53x_wp,
 label = cms.string("full53x")
 )

####################################################################################################################  
full_53x_chs = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("CondFormats/JetMETObjects/data/TMVAClassificationCategory_JetID_53X_chs_Dec2012.weights.xml"),
 #tmvaWeights = cms.string("RecoJets/JetProducers/data/TMVAClassificationCategory_JetID_53X_chs_Dec2012.weights.xml"),
 tmvaMethod  = cms.string("JetIDMVAHighPt"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "nvtx"     ,
    "dZ"       , 
    "beta"     , 
    "betaStar" , 
    "nCharged" , 
    "nNeutrals", 
    "dR2Mean"  , 
    "ptD"      , 
    "frac01"   , 
    "frac02"   , 
    "frac03"   , 
    "frac04"   , 
    "frac05"   , 
    ),
 tmvaSpectators = cms.vstring(
    "jetPt",
    "jetEta",
    "jetPhi"
    ),
 JetIdParams = full_53x_chs_wp,
 label = cms.string("full")
 )
####################################################################################################################  
met_53x = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("RecoJets/JetProducers/data/TMVAClassificationCategory_JetID_MET_53X_Dec2012.weights.xml.gz"),
 tmvaMethod  = cms.string("JetIDMVAMET"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "nvtx"     ,
    "jetPt"    ,
    "jetEta"   ,
    "jetPhi"   ,
    "dZ"       , 
    "beta"     , 
    "betaStar" , 
    "nCharged" , 
    "nNeutrals", 
    "dR2Mean"  , 
    "ptD"      , 
    "frac01"   , 
    "frac02"   , 
    "frac03"   , 
    "frac04"   , 
    "frac05"   , 
    ),
 tmvaSpectators = cms.vstring(),
 JetIdParams = met_53x_wp,
 label = cms.string("met53x")
 )
####################################################################################################################  
full_5x = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("RecoJets/JetProducers/data/TMVAClassification_5x_BDT_fullPlusRMS.weights.xml.gz"),
 tmvaMethod  = cms.string("BDT_fullPlusRMS"),
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
 JetIdParams = full_5x_wp,
 label = cms.string("full")
 )

####################################################################################################################  
simple_5x = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("RecoJets/JetProducers/data/TMVAClassification_5x_BDT_simpleNoVtxCat.weights.xml.gz"),
 tmvaMethod  = cms.string("BDT_simpleNoVtxCat"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "frac01",
    "frac02",
    "frac03",
    "frac04",
    "frac05",
    "nvtx",
    "beta",
    "betaStar",
    ),
 tmvaSpectators = cms.vstring(
    "jetPt",
    "jetEta",
    ),
 JetIdParams = simple_5x_wp,
 label = cms.string("simple")
 )

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
simple_5x_chs = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("RecoJets/JetProducers/data/TMVAClassification_5x_BDT_chsSimpleNoVtxCat.weights.xml.gz"),
 tmvaMethod  = cms.string("BDT_chsSimpleNoVtxCat"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "frac01",
    "frac02",
    "frac03",
    "frac04",
    "frac05",
    "nvtx",
    "beta",
    "betaStar",
    ),
 tmvaSpectators = cms.vstring(
    "jetPt",
    "jetEta",
    ),
 JetIdParams = simple_5x_chs_wp,
 label = cms.string("simple")
 )

####################################################################################################################  
full = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("RecoJets/JetProducers/data/TMVAClassification_PuJetIdOptMVA.weights.xml.gz"),
 tmvaMethod  = cms.string("PuJetIdOptMVA"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "frac01",
    "frac02",
    "frac03",
    "frac04",
    "frac05",
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
 JetIdParams = PuJetIdOptMVA_wp,
 label = cms.string("full")
 )

####################################################################################################################  
simple = cms.PSet( 
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("RecoJets/JetProducers/data/TMVAClassification_PuJetIdMinMVA.weights.xml.gz"),
 tmvaMethod  = cms.string("PuJetIdMinMVA"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "frac01",
    "frac02",
    "frac03",
    "frac04",
    "frac05",
    "beta",
    "betaStar",
    ),
 tmvaSpectators = cms.vstring(
    "nvtx",
    "jetPt",
    "jetEta",
    ),
 JetIdParams = PuJetIdMinMVA_wp,
 label = cms.string("simple")
 )
####################################################################################################################  
cutbased = cms.PSet( 
 impactParTkThreshold = cms.double(1.),
 cutBased = cms.bool(True),
 JetIdParams = PuJetIdCutBased_wp,
 label = cms.string("cutbased")
 )

####################################################################################################################  
PhilV0 = cms.PSet( 
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("RecoJets/JetProducers/data/mva_JetID.weights.xml.gz"),
 tmvaMethod  = cms.string("JetID"),
 version = cms.int32(0),
 JetIdParams = EmptyJetIdParams
)


####################################################################################################################  
PhilV1 = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("RecoJets/JetProducers/data/mva_JetID_v1.weights.xml.gz"),
 tmvaMethod  = cms.string("JetID"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "nvtx",
    "jetPt",
    "jetEta",
    "jetPhi",
    "dZ",
    "d0",
    "beta",
    "betaStar",
    "nCharged",
    "nNeutrals",
    "dRMean",
    "frac01",
    "frac02",
    "frac03",
    "frac04",
    "frac05",
    ),
 tmvaSpectators = cms.vstring(),
 JetIdParams = JetIdParams,
 label = cms.string("philv1")
 )

