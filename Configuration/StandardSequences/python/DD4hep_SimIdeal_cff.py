import FWCore.ParameterSet.Config as cms

# CMSSW/Geant4 interface
from SimG4Core.Configuration.DD4hep_SimG4Core_cff import *

psimTask = cms.Task(cms.TaskPlaceholder("randomEngineStateProducer"), g4SimHits)
psim = cms.Sequence(psimTask)
# foo bar baz
# Nw3YlDilss6nY
# 2nwAKZpEVJ6E5
