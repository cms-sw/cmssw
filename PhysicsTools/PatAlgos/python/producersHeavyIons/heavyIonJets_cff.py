import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import *
from RecoHI.HiJetAlgos.HiGenCleaner_cff import *
heavyIonCleaned = cms.Sequence(genPartons*hiPartons+heavyIonCleanedGenJets)

makeHeavyIonJets = cms.Sequence(
    patJetCorrections *

    # pat and HI specifics
    heavyIonCleaned *
    patJetGenJetMatch *
    patJetPartonMatch *

    # object production
    patJets
    )


