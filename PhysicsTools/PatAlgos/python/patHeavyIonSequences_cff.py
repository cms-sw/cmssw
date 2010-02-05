import FWCore.ParameterSet.Config as cms

# make heavyIonObjects
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonObjects_cff import *

# make selectedLayer1Objects
from PhysicsTools.PatAlgos.selectionLayer1.selectedLayer1Objects_cff import *

patHeavyIonDefaultSequence = cms.Sequence(
    heavyIonObjects * 
    selectedLayer1Jets *
    selectedLayer1Muons *
    selectedLayer1Photons
)
