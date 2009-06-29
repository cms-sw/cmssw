import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfPileUp_cfi  import *
from PhysicsTools.PFCandProducer.TopProjectors.pfNoPileUp_cfi import *

pfNoPileUpSequence = cms.Sequence(
    pfPileUp +
    pfNoPileUp 
    )
