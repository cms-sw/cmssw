import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import *
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonGenJetCleaner_cfi import *

makeHeavyIonJets = cms.Sequence(
    # reco pre-production
    jetCorrFactors *
    # pat and HI specifics
    heavyIonCleanedGenJets *
    jetGenJetMatch *
    # object production
    allLayer1Jets
    )


