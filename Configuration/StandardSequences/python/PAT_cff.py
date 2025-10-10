import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff import *
from PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff import *
from PhysicsTools.PatAlgos.slimming.slimming_cff import *
from RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi import *

patTask = cms.Task(
    patCandidatesTask,
    selectedPatCandidatesTask,
    slimmingTask,
    bunchSpacingProducer
)

from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllData
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC

## include cms.Path defined from event filters
from PhysicsTools.PatAlgos.slimming.metFilterPaths_cff import *
