import FWCore.ParameterSet.Config as cms

# the only thing FastSim runs from L1Reco is l1extraParticles
from L1Trigger.Configuration.L1Reco_cff import l1extraParticles

# If the Stage 1 trigger is running, there is also some different configuration.
# Note that this next file does nothing if the stage1L1Trigger era is not active, so
# it is safe to import even if the Stage 1 trigger is not required. It *MUST* be
# imported into this namespace, i.e. "from <module> import *".
from L1Trigger.Configuration.ConditionalStage1Configuration_cff import *

# must be set to true when used in HLT, as is the case for FastSim
l1extraParticles.centralBxOnly = True

L1Reco = cms.Sequence(l1extraParticles)
