import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonJets_cff import *
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonMuons_cff import *
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonPhotons_cff import *
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonProducer_cfi import *

heavyIonPatCandidates = cms.Sequence(
    heavyIon +
    makeHeavyIonJets +
    makeHeavyIonPhotons +
    makeHeavyIonMuons
)
