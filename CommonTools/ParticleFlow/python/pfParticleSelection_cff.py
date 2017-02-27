import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfNoPileUpIso_cff  import *
from CommonTools.ParticleFlow.pfNoPileUpJME_cff  import *
from CommonTools.ParticleFlow.pfNoPileUp_cff  import *
from CommonTools.ParticleFlow.ParticleSelectors.pfSortByType_cff import *

pfParticleSelectionTask = cms.Task(
    pfNoPileUpIsoTask,
    # In principle JME sequence should go here, but this is used in RECO
    # in addition to here, and is used in the "first-step" PF process
    # so needs to go later. 
    #pfNoPileUpJMETask,
    pfNoPileUpTask,
    pfSortByTypeTask 
    )
pfParticleSelectionSequence = cms.Sequence(pfParticleSelectionTask)
