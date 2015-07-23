import FWCore.ParameterSet.Config as cms
from RecoJets.JetProducers.hltPUIdParams_cfi import *

####################################################################################################################  
full_74x = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 tmvaWeights = cms.string("RecoJets/JetProducers/data/MVAJetPuID.weights.xml.gz"),
 tmvaMethod  = cms.string("BDTG"),
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
    "pull"   , 
    "jetR"   , 
    "jetRchg"	
    ),
 tmvaSpectators = cms.vstring(
    "jetEta",
    "jetPt",
    ),
 JetIdParams = full_74x_wp,
 label = cms.string("CATEv0")
 )

