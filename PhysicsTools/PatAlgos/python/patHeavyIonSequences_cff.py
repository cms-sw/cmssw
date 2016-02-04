import FWCore.ParameterSet.Config as cms

# make heavyIonPatCandidates
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonPatCandidates_cff import *

# make selectedLayer1Objects
from PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff import *

patHeavyIonDefaultSequence = cms.Sequence(
    heavyIonPatCandidates  * 
    selectedPatJets        *
    selectedPatMuons       *
    selectedPatPhotons
)
