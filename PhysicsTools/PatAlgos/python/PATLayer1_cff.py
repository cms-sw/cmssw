import FWCore.ParameterSet.Config as cms

# PAT Layer 1 default sequence
# build the Objects (Jets, Muons, Electrons, METs, Taus)
from PhysicsTools.PatAlgos.PATObjectProducers_cff import *
patLayer1 = cms.Sequence(allObjects)

