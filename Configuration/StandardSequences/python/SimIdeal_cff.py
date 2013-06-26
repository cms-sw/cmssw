import FWCore.ParameterSet.Config as cms

# CMSSW/Geant4 interface
from SimG4Core.Configuration.SimG4Core_cff import *

psim = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")*g4SimHits)
