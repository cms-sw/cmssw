import FWCore.ParameterSet.Config as cms

from L1Trigger.L1ExtraFromDigis.l1extraParticles_cff import *
# replaced with GT emulator for standard production
#include "L1Trigger/L1ExtraFromDigis/data/l1extraParticleMap.cfi"
L1Extra = cms.Sequence(l1extraParticles)

