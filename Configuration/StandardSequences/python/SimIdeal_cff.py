import FWCore.ParameterSet.Config as cms

# CMSSW/Geant4 interface
from SimG4Core.Configuration.SimG4Core_cff import *

psimTask = cms.Task(cms.TaskPlaceholder("randomEngineStateProducer"), g4SimHits)
psim = cms.Sequence(psimTask)
