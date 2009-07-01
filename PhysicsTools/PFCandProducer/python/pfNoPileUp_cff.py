import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfPileUp_cff  import *
from PhysicsTools.PFCandProducer.TopProjectors.noPileUp_cfi import *

pfNoPileUpSequence = cms.Sequence(
    pfPileUp +
    noPileUp 
    )
