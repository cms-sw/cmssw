import FWCore.ParameterSet.Config as cms

## setup the analyzer module
from PhysicsTools.PatExamples.PatJetAnalyzer_cfi import *

## setup the jet energy corrections for the reco jet
from JetMETCorrections.Configuration.JetCorrectionProducers_cff import *

###
## setup the configuration for Exercise 1 (c)
###

compareRaw = analyzePatJets.clone(corrLevel="raw", reco="ak5CaloJets")
compareL2  = analyzePatJets.clone(corrLevel="rel", reco="ak5CaloJetsL2")
compareL3  = analyzePatJets.clone(corrLevel="abs", reco="ak5CaloJetsL2L3")

## sequence for Exercise 1(c)
comparePatAndReco = cms.Sequence(
    ak5CaloJetsL2   *
    ak5CaloJetsL2L3 *
    compareRaw +
    compareL2  +
    compareL3
)
   
###
## setup the configuration for Exercise 1 (d)
###

calibRaw = analyzeJES.clone(corrLevel="raw")
calibL2  = analyzeJES.clone(corrLevel="rel")
calibL3  = analyzeJES.clone(corrLevel="abs")
calibL5  = analyzeJES.clone(corrLevel="had:uds")
calibL7  = analyzeJES.clone(corrLevel="part:uds")

## sequence for Exercise 1(d)
doJetResponse = cms.Sequence(
    ak5CaloJetsL2   *
    ak5CaloJetsL2L3 *
    calibRaw +
    calibL2  +
    calibL3  +
    calibL5  +
    calibL7
)
