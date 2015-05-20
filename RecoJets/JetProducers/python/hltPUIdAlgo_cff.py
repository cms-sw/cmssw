import FWCore.ParameterSet.Config as cms
from RecoJets.JetProducers.hltPUIdParams_cfi import *

####################################################################################################################  
full_74x = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("RecoJets/JetProducers/data/.weights.xml"),
 tmvaMethod  = cms.string("JetID"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "rho"     ,
    "nTot"     , 
    "nCh" , 
    "axisMajor" , 
    "axisMinor",	
    "fRing0",
    "fRing1",
    "fRing2",
    "fRing3",		 
    "ptD"      , 
    "beta"   , 
    "betaStar"   , 
    "DR_weighted"   , 
    "min(pull,0.1)"   , 
    "jetR"   , 
    "jetRchg"	
    ),
 tmvaSpectators = cms.vstring(
    "pt",
    "eta",
    ),
 JetIdParams = full_74x_wp,
 label = cms.string("CATEv0")
 )

