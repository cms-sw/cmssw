import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfPileUp_cfi  import *


pfPileUpSequence = cms.Sequence(
    pfPileUp 
    )

