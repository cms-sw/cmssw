import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatExamples.PatElectronAnalyzer_cfi import *

plainElectronID = analyzePatElectron.clone(mode=1, electronID = "none")
looseElectronID = analyzePatElectron.clone(mode=1, electronID = "eidRobustLoose")
tightElectronID = analyzePatElectron.clone(mode=1, electronID = "eidRobustTight")

tagAndProbeAnalysis = cms.Sequence(
    plainElectronID +
    looseElectronID +
    tightElectronID
    )
