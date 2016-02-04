import FWCore.ParameterSet.Config as cms

# CMSSW/Geant4 interface
from SimG4Core.Configuration.SimG4Core_cff import *

g4SimHits.NonBeamEvent = cms.bool(True)
g4SimHits.Generator.HepMCProductLabel = cms.string('generator')
g4SimHits.Generator.ApplyEtaCuts = cms.bool(False)

psim = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")*g4SimHits)


