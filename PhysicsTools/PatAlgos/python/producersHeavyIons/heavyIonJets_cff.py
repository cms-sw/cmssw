import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import *
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonGenJetCleaner_cfi import *

makeHeavyIonJets = cms.Sequence(
    # reco pre-production
#    jetCorrFactors * # changed by F.Blekman to fix compilation errors. Please check/confirm.
    patJetCorrections *
    # pat and HI specifics
    heavyIonCleanedGenJets *
#    jetGenJetMatch * and another change...
    patJetGenJetMatch *
    # object production
    patJets
    )


