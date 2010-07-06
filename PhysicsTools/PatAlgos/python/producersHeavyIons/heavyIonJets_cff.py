import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import *
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonGenJetCleaner_cfi import *

makeHeavyIonJets = cms.Sequence(
    patJetCorrections *

    # pat and HI specifics
    heavyIonCleaned *
    patJetGenJetMatch *
    patJetPartonMatch *

    # object production
    patJets
    )


