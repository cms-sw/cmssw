import FWCore.ParameterSet.Config as cms

# high level reco tasks needed before making PAT objects
from PhysicsTools.PatAlgos.recoLayer0.aodReco_cff import *

# MC matching: sequence patMCTruth
from PhysicsTools.PatAlgos.mcMatchLayer0.mcMatchSequences_cff   import  *

# make layer 1 objects: sequence allLayer1Objects
from PhysicsTools.PatAlgos.producersLayer1.allLayer1Objects_cff import *

# select layer 1 objects, make hemispheres: sequence selectedLayer1Objects
from PhysicsTools.PatAlgos.selectionLayer1.selectedLayer1Objects_cff import *

# clean layer 1 objects, make hemispheres: sequence cleanLayer1Objects
from PhysicsTools.PatAlgos.cleaningLayer1.cleanLayer1Objects_cff import *

# count selected layer 1 objects (including total number of leptons): sequence countLayer1Objects
from PhysicsTools.PatAlgos.selectionLayer1.countLayer1Objects_cff import *

# trigger info
from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cff import *

beforeLayer1Objects = cms.Sequence(
    patAODReco +  # use '+', as there is no dependency 
    patMCTruth    # among these sequences
)
#beforeLayer1Objects.doc = "Sequence to be run before producing PAT Objects"

patDefaultSequence = cms.Sequence(
      beforeLayer1Objects    # using '*', as the order is fixed.
    * allLayer1Objects
    * selectedLayer1Objects
    * cleanLayer1Objects 
    * countLayer1Objects
)
#patDefaultSequence.doc = "Default PAT Sequence from AOD to PAT Objects, including filters"
