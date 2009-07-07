import FWCore.ParameterSet.Config as cms

from DQM.Physics.qcdPhotonsDQM_cfi import *
from DQM.Physics.bphysicsOniaDQM_cfi import *
from DQM.Physics.qcdHighPtDQM_cfi import *

dqmPhysics = cms.Sequence(qcdPhotonsDQM*qcdHighPtDQM*bphysicsOniaDQM)

