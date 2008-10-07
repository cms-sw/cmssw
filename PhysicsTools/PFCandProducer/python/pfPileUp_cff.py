import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfPileUp_cfi  import *
from PhysicsTools.PFCandProducer.pfNoPileUp_cfi import *

pfPileUpSequence = cms.Sequence(
    pfPileUp+
    pfNoPileUp
    )

