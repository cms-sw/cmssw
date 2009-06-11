import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatExamples.PatElectronAnalyzer_cfi import *

plainElectronID = analyzePatElectron.clone(mode=1)
looseElectronID = analyzePatElectron.clone(mode=1)
tightElectronID = analyzePatElectron.clone(mode=1)

plainElectronID.tagAndProbeMode.electronID = "none"
looseElectronID.tagAndProbeMode.electronID = "eidRobustLoose"
tightElectronID.tagAndProbeMode.electronID = "eidRobustTight"

tagAndProbeAnalysis = cms.Sequence(
    plainElectronID +
    looseElectronID +
    tightElectronID
    )
